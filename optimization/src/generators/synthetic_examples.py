"""Online Synthetic Example Generator.

Based on CHASE-SQL Section 3.3.
Dynamically generates relevant few-shot examples for each test question.
Step 1: Generate generic examples based on SQL features (JOIN, GROUP BY, etc.)
Step 2: Generate targeted examples based on relevant columns.
"""

import random
import re
from pathlib import Path

from openai import AsyncOpenAI

PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent.parent / "prompts" / "synthetic_gen.txt").read_text()

EXAMPLE_GEN_PROMPT = """Given the following database schema, generate {n} diverse example question-SQL pairs that demonstrate these SQL patterns: {patterns}.

The examples should:
- Use the exact table and column names from the schema
- Be realistic questions a user might ask
- Cover different aspects of the patterns listed
- Be diverse in complexity

### Database Schema:
{schema}

Generate {n} examples in this exact format:
Q: [question]
SQL: [query]

Examples:"""


def _detect_sql_patterns(question: str, evidence: str) -> list[str]:
    """Detect likely SQL patterns from the question text."""
    text = (question + " " + evidence).lower()
    patterns = []

    if any(w in text for w in ("each", "per", "by", "group", "average", "total", "count", "sum")):
        patterns.append("GROUP BY with aggregation")
    if any(w in text for w in ("join", "from", "relate", "connect", "link", "both", "and the")):
        patterns.append("JOIN between tables")
    if any(w in text for w in ("most", "highest", "lowest", "maximum", "minimum", "top", "best", "worst")):
        patterns.append("ORDER BY with LIMIT")
    if any(w in text for w in ("not", "never", "without", "except", "exclude")):
        patterns.append("NOT IN or NOT EXISTS subquery")
    if any(w in text for w in ("more than", "less than", "greater", "at least", "at most")):
        patterns.append("HAVING or WHERE with comparison")
    if any(w in text for w in ("percentage", "ratio", "proportion", "rate", "fraction")):
        patterns.append("Division and CAST for percentages")
    if any(w in text for w in ("difference", "between", "compare", "versus")):
        patterns.append("Self-JOIN or CASE WHEN")

    if not patterns:
        patterns = ["SELECT with WHERE filter"]

    return patterns


def _parse_examples(content: str) -> str:
    """Parse generated examples from LLM output into formatted string."""
    examples = []
    # Match Q: ... SQL: ... pairs
    pairs = re.findall(
        r"Q:\s*(.+?)\s*(?:\n\s*)?SQL:\s*(.+?)(?=\nQ:|\Z)",
        content,
        re.DOTALL,
    )
    for q, sql in pairs:
        q = q.strip()
        sql = sql.strip().rstrip(";")
        if q and sql and sql.upper().startswith(("SELECT", "WITH")):
            examples.append(f"Q: {q}\nSQL: {sql}")

    return "\n\n".join(examples) if examples else ""


async def generate_examples(
    client: AsyncOpenAI,
    model: str,
    schema: str,
    question: str,
    evidence: str = "",
    n_examples: int = 3,
) -> str:
    """Generate synthetic few-shot examples relevant to the question."""
    patterns = _detect_sql_patterns(question, evidence)

    prompt = EXAMPLE_GEN_PROMPT.format(
        n=n_examples,
        patterns=", ".join(patterns),
        schema=schema,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        return _parse_examples(content)
    except Exception:
        return ""


def _shuffle_schema(schema: str) -> str:
    """Randomly shuffle table order in schema."""
    blocks = schema.strip().split("\n\n")
    random.shuffle(blocks)
    return "\n\n".join(blocks)


async def generate_candidates(
    client: AsyncOpenAI,
    model: str,
    schema: str,
    question: str,
    evidence: str = "",
    value_hint: str = "",
    n_candidates: int = 7,
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> list[str]:
    """Generate SQL candidates using online synthetic examples.

    First generates relevant few-shot examples, then uses them
    to generate multiple SQL candidates.
    """
    # Generate synthetic examples once per question
    examples = await generate_examples(client, model, schema, question, evidence)

    import asyncio

    async def _gen_one(i: int) -> str:
        shuffle = i > 0
        cur_schema = _shuffle_schema(schema) if shuffle else schema
        prompt = PROMPT_TEMPLATE.format(
            schema=cur_schema,
            question=question,
            evidence=evidence,
            value_hint=value_hint,
            examples=examples if examples else "(No examples available - generate the SQL directly based on the schema and question)",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.gather(*[_gen_one(i) for i in range(n_candidates)])
