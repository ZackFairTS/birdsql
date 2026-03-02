"""Extract database schema from SQLite files for prompt construction."""

import sqlite3
from pathlib import Path


def extract_schema(db_path: str, num_sample_rows: int = 3) -> str:
    """
    Extract schema from a SQLite database.

    Returns CREATE TABLE DDL + sample rows for each table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all CREATE TABLE statements
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
    tables = cursor.fetchall()

    parts = []
    for table_name, create_sql in tables:
        parts.append(create_sql.strip() + ";")

        if num_sample_rows > 0:
            try:
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {num_sample_rows}')
                rows = cursor.fetchall()
                if rows:
                    col_names = [desc[0] for desc in cursor.description]
                    # Truncate long values
                    formatted_rows = []
                    for row in rows:
                        formatted_row = []
                        for val in row:
                            s = str(val) if val is not None else "NULL"
                            if len(s) > 50:
                                s = s[:47] + "..."
                            formatted_row.append(s)
                        formatted_rows.append(formatted_row)

                    parts.append(f"/* Sample rows from {table_name}:")
                    parts.append(" | ".join(col_names))
                    for row in formatted_rows:
                        parts.append(" | ".join(row))
                    parts.append("*/")
            except Exception:
                pass  # Skip tables that can't be queried

        parts.append("")  # blank line between tables

    conn.close()
    return "\n".join(parts)


class SchemaCache:
    """Cache schemas by db_id to avoid re-reading SQLite files."""

    def __init__(self, db_root_path: str, num_sample_rows: int = 3):
        self.db_root = Path(db_root_path)
        self.num_sample_rows = num_sample_rows
        self._cache: dict[str, str] = {}

    def get(self, db_id: str) -> str:
        if db_id not in self._cache:
            db_path = self.db_root / db_id / f"{db_id}.sqlite"
            self._cache[db_id] = extract_schema(str(db_path), self.num_sample_rows)
        return self._cache[db_id]
