"""Extract SQL from model output."""

import re

SQL_KEYWORDS = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")


def parse_sql(content: str) -> str:
    """
    Extract SQL from model output. Handles:
    1. Clean SQL (just the query)
    2. Markdown code blocks
    3. SQL mixed with explanatory text
    """
    if not content or not content.strip():
        return "SELECT 1"

    content = content.strip()

    # Try markdown code blocks first
    matches = re.findall(r"```(?:sql|sqlite|SQL)?\s*\n?(.*?)```", content, re.DOTALL | re.IGNORECASE)
    for match in matches:
        match = match.strip()
        if match and any(match.upper().startswith(kw) for kw in SQL_KEYWORDS):
            return _clean_sql(match)

    # Check if content itself is SQL (starts with keyword)
    if any(content.upper().lstrip().startswith(kw) for kw in SQL_KEYWORDS):
        # Take everything up to potential explanation (double newline or common markers)
        sql = re.split(r"\n\n(?![\s]*(?:FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|UNION|INTERSECT|EXCEPT|AND|OR|ON|SET|VALUES|INTO))",
                       content, maxsplit=1)[0]
        return _clean_sql(sql)

    # Try to find SQL starting with keyword anywhere in text
    sql_match = re.search(
        r"((?:SELECT|WITH)\b.*?)(?:\n\n|\Z)",
        content, re.DOTALL | re.IGNORECASE,
    )
    if sql_match:
        return _clean_sql(sql_match.group(1))

    return _clean_sql(content)


def _clean_sql(sql: str) -> str:
    """Clean up extracted SQL."""
    sql = sql.strip()
    sql = sql.rstrip(";")
    # Remove trailing explanation lines that don't look like SQL
    lines = sql.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines at the end
        if not stripped and cleaned:
            # Keep internal blank lines in case of formatted SQL
            cleaned.append(line)
            continue
        # Skip lines that look like natural language explanation (not SQL)
        if stripped.startswith(("--", "/*", "Note:", "This ", "The ", "Here ", "Explanation")):
            if stripped.startswith(("--", "/*")):
                # SQL comments - keep them
                cleaned.append(line)
            continue
        cleaned.append(line)
    # Remove trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip().rstrip(";") if cleaned else "SELECT 1"
