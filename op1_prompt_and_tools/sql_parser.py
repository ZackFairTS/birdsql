"""Extract final SQL from the agentic conversation with multiple fallbacks."""

import json
import re
import sys
from pathlib import Path

# Import baseline parser as ultimate fallback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sql_parser import parse_sql as baseline_parse_sql


def extract_sql(messages: list[dict]) -> str:
    """Extract the final SQL from an agentic conversation.

    Fallback chain:
      1. <final_answer><sql_code>...</sql_code></final_answer>
      2. <sql_code>...</sql_code> alone
      3. SQL from last sqlite_query tool call args
      4. Baseline parser on last assistant message
      5. "SELECT 1"
    """
    # Collect all assistant messages (newest first)
    assistant_msgs = [
        m["content"] for m in reversed(messages)
        if m.get("role") == "assistant" and m.get("content")
    ]

    full_text = "\n".join(assistant_msgs)

    # Fallback 1: <final_answer><sql_code>...</sql_code></final_answer>
    sql = _extract_final_answer_sql(full_text)
    if sql:
        return _clean(sql)

    # Fallback 2: <sql_code>...</sql_code> alone (last occurrence)
    sql = _extract_sql_code_tag(full_text)
    if sql:
        return _clean(sql)

    # Fallback 3: SQL from last sqlite_query tool call
    sql = _extract_from_tool_call(assistant_msgs)
    if sql:
        return _clean(sql)

    # Fallback 4: baseline parser on last assistant message
    if assistant_msgs:
        sql = baseline_parse_sql(assistant_msgs[0])
        if sql and sql != "SELECT 1":
            return sql

    # Fallback 5
    return "SELECT 1"


def _extract_final_answer_sql(text: str) -> str | None:
    """Extract SQL from <final_answer><sql_code>...</sql_code></final_answer>."""
    m = re.search(
        r"<final_answer>\s*<sql_code>\s*(.*?)\s*</sql_code>\s*</final_answer>",
        text, re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _extract_sql_code_tag(text: str) -> str | None:
    """Extract SQL from the last <sql_code>...</sql_code> tag."""
    matches = re.findall(r"<sql_code>\s*(.*?)\s*</sql_code>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _extract_from_tool_call(assistant_msgs: list[str]) -> str | None:
    """Extract SQL from the last sqlite_query tool call arguments."""
    tc_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for msg in assistant_msgs:
        for m in reversed(list(tc_re.finditer(msg))):
            try:
                outer = json.loads(m.group(1).strip())
                # Unwrap router_tools envelope
                if outer.get("name") == "router_tools":
                    inner = outer.get("arguments", {})
                    if inner.get("name") == "sqlite_query":
                        sql = inner.get("args", {}).get("sql", "")
                        if sql.strip():
                            return sql.strip()
                elif outer.get("name") == "sqlite_query":
                    sql = outer.get("arguments", outer.get("args", {})).get("sql", "")
                    if sql.strip():
                        return sql.strip()
            except (json.JSONDecodeError, AttributeError):
                continue
    return None


def _clean(sql: str) -> str:
    """Strip trailing semicolons and whitespace."""
    sql = sql.strip().rstrip(";").strip()
    return sql if sql else "SELECT 1"
