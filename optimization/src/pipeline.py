"""End-to-end CHASE-SQL pipeline.

Orchestrates: Value Retrieval → 3 Generators → Query Fixer → Selection
"""

import asyncio
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

# Ensure project root is on path for base module imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import base project modules (birdsql/src/)
_base_schema = importlib.import_module("src.schema_extractor")
SchemaCache = _base_schema.SchemaCache

_base_parser = importlib.import_module("src.sql_parser")
parse_sql = _base_parser.parse_sql

from .generators import divide_conquer, query_plan, synthetic_examples
from .query_fixer import fix_all_candidates
from .selector import (
    TrainedSelector,
    select_by_consistency,
    select_by_consistency_with_empty_penalty,
    select_by_pairwise_llm,
)
from .value_retrieval import ValueRetriever


@dataclass
class PipelineConfig:
    """Configuration for the CHASE-SQL pipeline."""
    # Model settings
    model: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    api_base: str = "http://localhost:8000/v1"

    # Generation settings
    n_candidates_per_generator: int = 7
    temperature: float = 0.8
    max_tokens: int = 4096

    # Feature flags
    enable_value_retrieval: bool = True
    enable_divide_conquer: bool = True
    enable_query_plan: bool = True
    enable_synthetic_examples: bool = True
    enable_query_fixer: bool = True
    fixer_max_iterations: int = 3

    # Selection method: "consistency", "consistency_empty_penalty", "pairwise_llm", "trained"
    selection_method: str = "consistency"
    trained_selector_path: str | None = None

    # Database paths
    db_root_path: str = "data/dev_databases"

    # Concurrency
    concurrency: int = 4  # Per-question internal concurrency


@dataclass
class PipelineResult:
    """Result from processing a single question."""
    question_id: int
    db_id: str
    question: str
    selected_sql: str
    all_candidates: list[str] = field(default_factory=list)
    fixed_candidates: list[str] = field(default_factory=list)
    selection_stats: dict = field(default_factory=dict)
    value_hint: str = ""
    generator_counts: dict = field(default_factory=dict)


class ChaseSQLPipeline:
    """Main CHASE-SQL pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = AsyncOpenAI(base_url=config.api_base, api_key="dummy")
        self.schema_cache = SchemaCache(config.db_root_path)
        self.value_retriever = ValueRetriever(config.db_root_path) if config.enable_value_retrieval else None
        self.trained_selector = None
        if config.trained_selector_path:
            self.trained_selector = TrainedSelector(config.trained_selector_path)

    async def process_single(self, entry: dict, idx: int) -> PipelineResult:
        """Process a single question through the full pipeline."""
        db_id = entry["db_id"]
        question = entry["question"]
        evidence = entry.get("evidence", "")
        schema = self.schema_cache.get(db_id)
        db_path = str(Path(self.config.db_root_path) / db_id / f"{db_id}.sqlite")

        # Step 1: Value Retrieval
        value_hint = ""
        if self.value_retriever:
            try:
                value_hint = await self.value_retriever.retrieve(
                    self.client, self.config.model, question, evidence, db_id
                )
            except Exception:
                pass

        # Step 2: Generate candidates from all enabled generators
        generator_tasks = []
        generator_names = []

        if self.config.enable_divide_conquer:
            generator_tasks.append(
                divide_conquer.generate_candidates(
                    self.client, self.config.model, schema, question, evidence,
                    value_hint, self.config.n_candidates_per_generator,
                    self.config.temperature, self.config.max_tokens,
                )
            )
            generator_names.append("divide_conquer")

        if self.config.enable_query_plan:
            generator_tasks.append(
                query_plan.generate_candidates(
                    self.client, self.config.model, schema, question, evidence,
                    value_hint, self.config.n_candidates_per_generator,
                    self.config.temperature, self.config.max_tokens,
                )
            )
            generator_names.append("query_plan")

        if self.config.enable_synthetic_examples:
            generator_tasks.append(
                synthetic_examples.generate_candidates(
                    self.client, self.config.model, schema, question, evidence,
                    value_hint, self.config.n_candidates_per_generator,
                    self.config.temperature, self.config.max_tokens,
                )
            )
            generator_names.append("synthetic_examples")

        if not generator_tasks:
            return PipelineResult(
                question_id=idx, db_id=db_id, question=question,
                selected_sql="SELECT 1",
            )

        # Run generators concurrently
        raw_results = await asyncio.gather(*generator_tasks, return_exceptions=True)

        # Collect all candidates and parse SQL
        all_candidates = []
        generator_counts = {}
        for name, result in zip(generator_names, raw_results):
            if isinstance(result, Exception):
                generator_counts[name] = 0
                continue
            sqls = [parse_sql(r) for r in result if not r.startswith("ERROR:")]
            generator_counts[name] = len(sqls)
            all_candidates.extend(sqls)

        # Filter out SELECT 1 fallbacks from parse failures
        all_candidates = [s for s in all_candidates if s.strip().upper() != "SELECT 1"]

        if not all_candidates:
            # Fallback: try a simple baseline prompt (temperature=0, no CoT)
            fallback_sql = await self._baseline_fallback(schema, question, evidence)
            return PipelineResult(
                question_id=idx, db_id=db_id, question=question,
                selected_sql=fallback_sql,
                generator_counts=generator_counts,
            )

        # Step 3: Query Fixer
        if self.config.enable_query_fixer:
            fixed_candidates = await fix_all_candidates(
                self.client, self.config.model, all_candidates,
                db_path, schema, question, evidence,
                self.config.fixer_max_iterations,
            )
        else:
            fixed_candidates = all_candidates

        # Filter SELECT 1 from fixed candidates
        fixed_candidates = [s for s in fixed_candidates if s.strip().upper() != "SELECT 1"]
        if not fixed_candidates:
            fallback_sql = await self._baseline_fallback(schema, question, evidence)
            return PipelineResult(
                question_id=idx, db_id=db_id, question=question,
                selected_sql=fallback_sql,
                all_candidates=all_candidates,
                generator_counts=generator_counts,
            )

        # Step 4: Selection
        selected_sql, stats = await self._select(
            fixed_candidates, schema, question, evidence, db_path
        )

        return PipelineResult(
            question_id=idx,
            db_id=db_id,
            question=question,
            selected_sql=selected_sql,
            all_candidates=all_candidates,
            fixed_candidates=fixed_candidates,
            selection_stats=stats,
            value_hint=value_hint,
            generator_counts=generator_counts,
        )

    async def _select(
        self,
        candidates: list[str],
        schema: str,
        question: str,
        evidence: str,
        db_path: str,
    ) -> tuple[str, dict]:
        """Select the best SQL from candidates using configured method."""
        method = self.config.selection_method

        if method == "trained" and self.trained_selector and self.trained_selector.is_available():
            return self.trained_selector.select_best(candidates, schema, question, evidence)
        elif method == "pairwise_llm":
            return await select_by_pairwise_llm(
                self.client, self.config.model, candidates, schema, question, evidence, db_path
            )
        elif method == "consistency_empty_penalty":
            return select_by_consistency_with_empty_penalty(candidates, db_path)
        else:
            # Default: self-consistency
            return select_by_consistency(candidates, db_path)

    async def _baseline_fallback(self, schema: str, question: str, evidence: str) -> str:
        """Simple baseline prompt as fallback when all generators fail."""
        _base_prompt = importlib.import_module("src.prompt_builder")
        messages = _base_prompt.build_prompt(schema, question, evidence)
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            sql = parse_sql(content)
            return sql if sql.strip().upper() != "SELECT 1" else "SELECT 1"
        except Exception:
            return "SELECT 1"

    async def process_batch(
        self,
        entries: list[dict],
        concurrency: int = 4,
        progress_callback=None,
    ) -> list[PipelineResult]:
        """Process a batch of questions with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)
        results = [None] * len(entries)

        async def _process(i: int, entry: dict):
            async with semaphore:
                result = await self.process_single(entry, i)
                results[i] = result
                if progress_callback:
                    progress_callback(i, result)
                return result

        tasks = [_process(i, entry) for i, entry in enumerate(entries)]
        await asyncio.gather(*tasks)

        return results
