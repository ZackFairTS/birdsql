"""Extract M-Schema representation from SQLite databases.

M-Schema format (per XGenerationLab/M-Schema):

【DB_ID】 db_name
【Schema】
# Table: table_name
[
(col1:TYPE, comment, Primary Key, Examples: [v1, v2, v3]),
(col2:TYPE, Examples: [v1, v2])
]
【Foreign keys】
table1.col1=table2.col2
"""

import re
import sqlite3
from pathlib import Path


def _fetch_distinct_examples(cursor, table_name: str, col_name: str,
                             max_num: int = 5, max_char_len: int = 50) -> list[str]:
    """Fetch up to max_num distinct non-null example values for a column."""
    try:
        cursor.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
            f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' '
            f'LIMIT {max_num}'
        )
        rows = cursor.fetchall()
    except Exception:
        return []

    examples = []
    for (val,) in rows:
        s = str(val)
        # Skip URLs and emails
        if 'http://' in s or 'https://' in s or '@' in s:
            return []
        if len(s) > max_char_len:
            # For long strings, keep only first example truncated
            if not examples:
                examples.append(s[:max_char_len])
            break
        examples.append(s)
    return examples


def _parse_column_comment(create_sql: str, col_name: str) -> str:
    """Extract inline comment from DDL (e.g., -- comment after column def)."""
    # Match: `col_name` ... -- comment  or  col_name ... -- comment
    pattern = rf'[`"\[]?{re.escape(col_name)}[`"\]]?\s+[^,\n]*?--\s*(.+?)(?:,|\n|\))'
    m = re.search(pattern, create_sql, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_mschema(db_path: str, db_id: str, num_examples: int = 3) -> str:
    """Generate M-Schema string for a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    )
    tables = cursor.fetchall()

    # Get foreign keys per table
    fk_lines = []
    for table_name, _ in tables:
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        fks = cursor.fetchall()
        for fk in fks:
            # fk: (id, seq, ref_table, from_col, to_col, ...)
            ref_table, from_col, to_col = fk[2], fk[3], fk[4]
            fk_lines.append(f"{table_name}.{from_col}={ref_table}.{to_col}")

    output = []
    output.append(f"【DB_ID】 {db_id}")
    output.append("【Schema】")

    for table_name, create_sql in tables:
        output.append(f"# Table: {table_name}")
        output.append("[")

        # Get column info via PRAGMA
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        # columns: (cid, name, type, notnull, dflt_value, pk)

        field_lines = []
        for col in columns:
            cid, col_name, col_type, notnull, default_val, is_pk = col
            col_type_upper = col_type.split("(")[0].upper() if col_type else ""

            parts = [f"{col_name}:{col_type_upper}"]

            # Comment from DDL
            comment = _parse_column_comment(create_sql or "", col_name)
            if comment:
                parts.append(comment)

            # Primary key
            if is_pk:
                parts.append("Primary Key")

            # Examples
            if num_examples > 0:
                examples = _fetch_distinct_examples(cursor, table_name, col_name, num_examples)
                if examples:
                    ex_str = ", ".join(examples)
                    parts.append(f"Examples: [{ex_str}]")

            field_lines.append("(" + ", ".join(parts) + ")")

        output.append(",\n".join(field_lines))
        output.append("]")

    # Foreign keys
    if fk_lines:
        output.append("【Foreign keys】")
        output.extend(fk_lines)

    conn.close()
    return "\n".join(output)


class MSchemaCache:
    """Cache M-Schema strings by db_id."""

    def __init__(self, db_root_path: str, num_examples: int = 3):
        self.db_root = Path(db_root_path)
        self.num_examples = num_examples
        self._cache: dict[str, str] = {}

    def get(self, db_id: str) -> str:
        if db_id not in self._cache:
            db_path = self.db_root / db_id / f"{db_id}.sqlite"
            self._cache[db_id] = extract_mschema(str(db_path), db_id, self.num_examples)
        return self._cache[db_id]
