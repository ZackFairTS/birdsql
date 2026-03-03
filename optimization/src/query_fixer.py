"""SQL Query Fixer: self-repair loop for generated SQL.

Based on CHASE-SQL Section 3.4.
Executes SQL, checks for errors/empty results, feeds errors back to LLM for repair.
Maximum β=3 iterations.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout
from openai import AsyncOpenAI

PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent / "prompts" / "query_fixer.txt").read_text()

# Import parse_sql from the base project (birdsql/src/)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
parse_sql = importlib.import_module("src.sql_parser").parse_sql


def execute_sql_check(db_path: str, sql: str, timeout: int = 10) -> tuple[bool, str, list | None]:
    """Execute SQL and check for errors.

    Returns (success, error_message, results).
    - success=True, error_message="", results=[...] on success
    - success=False, error_message="...", results=None on failure
    """
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result

    try:
        result = func_timeout(timeout, _exec)
        return True, "", result
    except FunctionTimedOut:
        return False, "Query timed out after 10 seconds. Simplify the query.", None
    except Exception as e:
        return False, str(e), None


async def fix_sql(
    client: AsyncOpenAI,
    model: str,
    sql: str,
    error: str,
    schema: str,
    question: str,
    evidence: str = "",
    max_tokens: int = 4096,
) -> str:
    """Ask LLM to fix a broken SQL query given the error message."""
    prompt = PROMPT_TEMPLATE.format(
        schema=schema,
        question=question,
        evidence=evidence,
        sql=sql,
        error=error,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return parse_sql(content)
    except Exception:
        return sql  # Return original if fix fails


async def fix_candidate(
    client: AsyncOpenAI,
    model: str,
    sql: str,
    db_path: str,
    schema: str,
    question: str,
    evidence: str = "",
    max_iterations: int = 3,
) -> str:
    """Run the self-repair loop on a single SQL candidate.

    Tries to execute the SQL; if it fails, sends error to LLM for repair.
    Repeats up to max_iterations times.
    """
    current_sql = sql

    for iteration in range(max_iterations):
        success, error, results = execute_sql_check(db_path, current_sql)

        if success:
            # Check for empty results - could indicate a logic error
            if results is not None and len(results) == 0 and iteration == 0:
                # Try to fix empty result on first iteration only
                error = (
                    "The query executed successfully but returned NO results (empty set). "
                    "This likely indicates a logic error such as: wrong column comparison, "
                    "missing JOIN condition, overly restrictive WHERE clause, or wrong table. "
                    "Please review and fix the query."
                )
                fixed = await fix_sql(
                    client, model, current_sql, error, schema, question, evidence
                )
                if fixed != current_sql:
                    current_sql = fixed
                    continue
            return current_sql

        # SQL failed - try to fix
        current_sql = await fix_sql(
            client, model, current_sql, error, schema, question, evidence
        )

    return current_sql


async def fix_all_candidates(
    client: AsyncOpenAI,
    model: str,
    candidates: list[str],
    db_path: str,
    schema: str,
    question: str,
    evidence: str = "",
    max_iterations: int = 3,
) -> list[str]:
    """Run self-repair on all candidates.

    Returns list of fixed SQL strings (same length as input).
    """
    import asyncio

    tasks = [
        fix_candidate(client, model, sql, db_path, schema, question, evidence, max_iterations)
        for sql in candidates
    ]
    return await asyncio.gather(*tasks)
