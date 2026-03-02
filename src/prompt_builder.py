"""Build prompts for Text-to-SQL inference."""

SYSTEM_PROMPT = """You are an expert SQL assistant. Given a database schema and a question, generate a valid SQLite query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- Do not include markdown code fences or backticks
- Do not include any explanations or comments
- The query must be valid SQLite syntax
- Use the exact table and column names from the schema"""


def build_prompt(schema: str, question: str, evidence: str = "") -> list[dict]:
    """
    Build chat messages for the OpenAI-compatible API.

    Returns list of message dicts for the chat completions API.
    """
    user_parts = [f"### Database Schema:\n{schema}"]

    if evidence and evidence.strip():
        user_parts.append(f"### External Knowledge:\n{evidence}")

    user_parts.append(f"### Question:\n{question}")
    user_parts.append("### SQL:")

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
