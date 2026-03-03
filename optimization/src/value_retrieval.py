"""Value Retrieval: extract keywords from questions and fuzzy-match against database values.

Based on CHASE-SQL Section 3.1 - uses LLM to extract potential filter values
from the question, then matches them against actual database column values
using edit distance and containment matching.
"""

import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path

from openai import AsyncOpenAI

KEYWORD_EXTRACTION_PROMPT = """Given a question about a database, extract all potential filter values that might appear in WHERE clauses, HAVING clauses, or JOIN conditions.

These are concrete values like names, dates, numbers, codes, categories, or specific strings mentioned in the question or evidence.

Do NOT extract:
- Aggregation targets (e.g., "highest", "average", "count")
- Column names or table names
- General descriptors (e.g., "all", "each", "total")

Output ONLY the values, one per line. If no filter values found, output "NONE".

Question: {question}
Evidence: {evidence}

Values:"""


async def extract_keywords(
    client: AsyncOpenAI,
    model: str,
    question: str,
    evidence: str = "",
) -> list[str]:
    """Use LLM to extract potential filter value keywords from the question."""
    prompt = KEYWORD_EXTRACTION_PROMPT.format(question=question, evidence=evidence)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        if "NONE" in content.upper():
            return []
        keywords = [line.strip().strip('"').strip("'") for line in content.strip().split("\n")]
        return [kw for kw in keywords if kw and len(kw) >= 2]
    except Exception:
        return []


def get_column_values(db_path: str, max_distinct: int = 100, max_len: int = 100) -> dict[str, list[str]]:
    """Get distinct string/text values for each column in the database.

    Returns dict mapping "table.column" -> list of distinct string values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    tables = cursor.fetchall()

    column_values = {}
    for table_name, create_sql in tables:
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()
            for col in columns:
                col_name = col[1]
                col_type = (col[2] or "").upper()
                # Focus on text-like columns for value matching
                if any(t in col_type for t in ("TEXT", "VARCHAR", "CHAR", "BLOB", "")):
                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                            f'WHERE "{col_name}" IS NOT NULL AND typeof("{col_name}") = "text" '
                            f"LIMIT {max_distinct}"
                        )
                        values = [str(row[0]) for row in cursor.fetchall() if row[0] and len(str(row[0])) <= max_len]
                        if values:
                            column_values[f"{table_name}.{col_name}"] = values
                    except Exception:
                        pass
        except Exception:
            pass
    conn.close()
    return column_values


def fuzzy_match(keyword: str, values: list[str], threshold: float = 0.6) -> list[tuple[str, float]]:
    """Find fuzzy matches for a keyword in a list of values.

    Uses both edit distance ratio and containment matching.
    Returns list of (value, score) tuples sorted by score descending.
    """
    matches = []
    keyword_lower = keyword.lower()
    for val in values:
        val_lower = val.lower()
        # Exact match
        if keyword_lower == val_lower:
            matches.append((val, 1.0))
            continue
        # Containment match
        if keyword_lower in val_lower or val_lower in keyword_lower:
            score = min(len(keyword_lower), len(val_lower)) / max(len(keyword_lower), len(val_lower))
            score = max(score, 0.8)  # Containment is strong signal
            matches.append((val, score))
            continue
        # Edit distance ratio
        ratio = SequenceMatcher(None, keyword_lower, val_lower).ratio()
        if ratio >= threshold:
            matches.append((val, ratio))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


class ValueRetriever:
    """Retrieves matching database values for question keywords."""

    def __init__(self, db_root_path: str):
        self.db_root = Path(db_root_path)
        self._value_cache: dict[str, dict[str, list[str]]] = {}

    def _get_values(self, db_id: str) -> dict[str, list[str]]:
        if db_id not in self._value_cache:
            db_path = str(self.db_root / db_id / f"{db_id}.sqlite")
            self._value_cache[db_id] = get_column_values(db_path)
        return self._value_cache[db_id]

    async def retrieve(
        self,
        client: AsyncOpenAI,
        model: str,
        question: str,
        evidence: str,
        db_id: str,
        top_k: int = 5,
    ) -> str:
        """Extract keywords and find matching database values.

        Returns a formatted string of matched values to inject into prompts.
        """
        keywords = await extract_keywords(client, model, question, evidence)
        if not keywords:
            return ""

        column_values = self._get_values(db_id)
        all_matches = []

        for keyword in keywords:
            for col_key, values in column_values.items():
                matches = fuzzy_match(keyword, values, threshold=0.6)
                for val, score in matches[:3]:
                    all_matches.append((keyword, col_key, val, score))

        if not all_matches:
            return ""

        # Deduplicate and sort by score
        seen = set()
        unique_matches = []
        for keyword, col_key, val, score in sorted(all_matches, key=lambda x: x[3], reverse=True):
            key = (col_key, val)
            if key not in seen:
                seen.add(key)
                unique_matches.append((keyword, col_key, val, score))

        top_matches = unique_matches[:top_k]

        lines = ["### Matched Database Values:"]
        for keyword, col_key, val, score in top_matches:
            table, col = col_key.split(".", 1)
            lines.append(f'- Column `{table}`.`{col}` contains value "{val}" (matched keyword: "{keyword}")')

        return "\n".join(lines)
