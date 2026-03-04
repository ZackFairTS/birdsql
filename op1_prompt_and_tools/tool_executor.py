"""Parse <tool_call> tags from model output and execute via tools.py."""

import json
import re
import traceback
from typing import Optional

from . import tools

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

TOOL_DISPATCH = {
    "sqlite_query": tools.sqlite_query,
    "sqlite_peek": tools.sqlite_peek,
    "bm25_search_sqlite": tools.bm25_search_sqlite,
}

MAX_RESULT_CHARS = 2000


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract the first <tool_call> JSON from model output.

    Handles the router_tools envelope:
        {"name":"router_tools","arguments":{"name":"<tool>","args":{...}}}

    Returns dict with keys 'tool_name' and 'tool_args', or None.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None

    raw = m.group(1).strip()
    try:
        outer = json.loads(raw)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing commas, single quotes
        cleaned = raw.replace("'", '"')
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            outer = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    # Unwrap router_tools envelope
    if outer.get("name") == "router_tools":
        args = outer.get("arguments", {})
        tool_name = args.get("name", "")
        tool_args = args.get("args", {})
    else:
        # Direct tool call (no envelope)
        tool_name = outer.get("name", "")
        tool_args = outer.get("arguments", outer.get("args", {}))

    if not tool_name:
        return None

    return {"tool_name": tool_name, "tool_args": tool_args}


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


async def execute_tool_call(text: str) -> Optional[str]:
    """Parse and execute a tool call from model output.

    Returns formatted <tool_result> string, or None if no tool call found.
    """
    parsed = parse_tool_call(text)
    if parsed is None:
        return None

    tool_name = parsed["tool_name"]
    tool_args = parsed["tool_args"]

    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        result_str = json.dumps({"error": f"unknown tool: {tool_name}"})
    else:
        try:
            result = await fn(**tool_args)
            result_str = json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            result_str = json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-500:],
            })

    result_str = _truncate(result_str)
    return f"<tool_result>\n{result_str}\n</tool_result>"
