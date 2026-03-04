# recipe/retool/generated_tools.py
# Async tool functions for VERL router.
# Exposed via your GeneratedToolsRouter (e.g., tool name "finance_data" or "router_tools")
# so the LLM can call (example):
# {"name":"finance_data","arguments":{"name":"sqlite_peek","args":{"db_id":"...", "sql":"SELECT ..."}}}

import os
import re
import sqlite3
import time
from typing import Dict, List, Tuple, Any, Optional
import os
import sqlite3
from typing import List, Dict
# add to the top-level __all__

# keep existing imports (os, re, sqlite3 already present)
from typing import Any  # if not already imported
import re

__all__ = ["sqlite_peek", "sqlite_query", "bm25_search_sqlite"]

# -------------------------
# Internals / guardrails
# -------------------------

_READONLY_SELECT_RX = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

def _roots_from_env() -> List[str]:
    """Read colon/OS-pathsep separated roots from BIRD_DB_ROOTS."""
    roots = os.environ.get("BIRD_DB_ROOTS", "")
    parts = [p for p in roots.split(os.pathsep) if p]
    return parts

def _candidate_names(db_id: str) -> List[str]:
    """Candidate filenames/paths tried under each root."""
    return [
        f"{db_id}.sqlite", f"{db_id}.sqlite3", f"{db_id}.db",
        os.path.join(db_id, f"{db_id}.sqlite"),
        os.path.join(db_id, f"{db_id}.sqlite3"),
        os.path.join(db_id, f"{db_id}.db"),
    ]

def _resolve_db_path(db_id: str) -> Optional[str]:
    """Resolve a DB path from env roots + db_id."""
    for r in _roots_from_env():
        for c in _candidate_names(db_id):
            p = os.path.join(r, c)
            if os.path.isfile(p):
                return p
    return None

def _ensure_select_only(sql: str) -> None:
    """Ensure SQL starts with SELECT and does not contain obvious mutating patterns."""
    if not isinstance(sql, str) or not _READONLY_SELECT_RX.match(sql):
        raise ValueError("Only read-only SELECT statements are allowed.")
    # cheap defense-in-depth against multiple statements or PRAGMA etc.
    lower = sql.lower()
    forbidden = [" pragma ", " attach ", " detach ", " insert ", " update ", " delete ", " drop ", " create ", " alter "]
    if any(tok in lower for tok in forbidden):
        raise ValueError("Only read-only SELECT statements are allowed (mutating/DDL tokens detected).")

def _connect_ro(db_path: str, busy_ms: int = 2000) -> sqlite3.Connection:
    """Open a read-only SQLite connection with safe PRAGMAs."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=busy_ms / 1000.0)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    return conn

def _set_progress_guard(conn: sqlite3.Connection, hard_timeout_s: float, max_steps: int) -> None:
    """Abort long-running queries via VM progress handler and soft deadline."""
    soft_deadline = time.perf_counter() + max(0.75 * hard_timeout_s, min(0.5, hard_timeout_s))
    steps = 0
    def _progress() -> int:
        nonlocal steps
        steps += 1
        if (max_steps and steps > max_steps) or (time.perf_counter() > soft_deadline):
            return 1
        return 0
    conn.set_progress_handler(_progress, 1000)

def _run_select(
    db_path: str,
    sql: str,
    *,
    hard_timeout_s: float = 20.0,
    busy_ms: int = 20000,
    max_steps: int = 1_000_000,
    cap_rows: Optional[int] = None,
) -> Tuple[List[List[Any]], List[str]]:
    """Execute read-only SELECT with guardrails and optional fetch cap."""
    _ensure_select_only(sql)
    start = time.perf_counter()
    conn = _connect_ro(db_path, busy_ms=busy_ms)
    _set_progress_guard(conn, hard_timeout_s, max_steps)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if cap_rows is not None and cap_rows >= 0:
            rows_t = cur.fetchmany(cap_rows)
        else:
            rows_t = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        # Convert tuples to lists for JSON friendliness
        rows = [list(r) for r in rows_t]
        return rows, cols
    finally:
        try:
            conn.close()
        finally:
            # Best-effort soft wallclock guard; hard limit enforced via progress handler above.
            _ = (time.perf_counter() - start)

def _force_limit(sql: str, default_limit: int) -> str:
    """Append LIMIT if missing (naive but effective)."""
    if " limit " not in sql.lower():
        return sql.rstrip().rstrip(";") + f" LIMIT {int(default_limit)}"
    return sql

# -------------------------
# Public async API
# -------------------------

import asyncio
import math
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# expects your existing helpers:
# _resolve_db_path(db_id) -> str
# _force_limit(sql, limit) -> str
# _run_select(db_path, sql, hard_timeout_s, busy_ms, max_steps, cap_rows=None) -> (rows, cols)

def _is_number_like(x) -> bool:
    if x is None: 
        return False
    if isinstance(x, (int, float)): 
        return True
    try:
        float(str(x))
        return True
    except Exception:
        return False

def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _strish(x) -> str:
    if x is None:
        return ""
    s = str(x)
    # guard insanely long blobs
    return s if len(s) <= 10_000 else s[:10_000]

def _char_class_profile(s: str) -> Dict[str, Any]:
    # simple character-class summary for a string
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    has_space = any(c.isspace() for c in s)
    has_punct = any(not (c.isalnum() or c.isspace()) for c in s)
    all_digits = s.isdigit() and len(s) > 0
    # crude date-ish / iso-ish detector
    looks_date = False
    if len(s) in (8, 10, 19):
        if "-" in s or "/" in s or ":" in s:
            looks_date = True
    looks_json = (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))
    return {
        "has_upper": has_upper,
        "has_lower": has_lower,
        "has_digit": has_digit,
        "has_space": has_space,
        "has_punct": has_punct,
        "all_digits": all_digits,
        "looks_date_like": looks_date,
        "looks_json_like": looks_json,
    }

def _prefixes(s: str, lens=(2, 3, 4)) -> Dict[str, str]:
    out = {}
    for L in lens:
        out[f"prefix_{L}"] = s[:L] if len(s) >= L else s
    return out



def _profile_columns_from_sample(rows: List[Tuple[Any, ...]], cols: List[str], *, topk: int = 10) -> Dict[str, Any]:
    n = len(rows)
    col_profiles: Dict[str, Any] = {}
    # transpose columns cheaply
    col_data = defaultdict(list)
    for r in rows:
        for i, v in enumerate(r):
            col_data[cols[i]].append(v)

    for name in cols:
        data = col_data[name]
        nulls = sum(1 for v in data if v is None)
        non_null = n - nulls
        distinct_vals = set(v for v in data if v is not None)
        numeric_vals = []
        str_vals = []
        for v in data:
            if v is None:
                continue
            if _is_number_like(v):
                f = _maybe_float(v)
                if f is not None and not math.isnan(f) and not math.isinf(f):
                    numeric_vals.append(f)
            s = _strish(v)
            str_vals.append(s)

        # shape for numbers
        num_profile = None
        if numeric_vals:
            try:
                num_profile = {
                    "min": min(numeric_vals),
                    "max": max(numeric_vals),
                    "mean": statistics.fmean(numeric_vals) if len(numeric_vals) > 0 else None,
                    "median": statistics.median(numeric_vals) if len(numeric_vals) > 0 else None,
                }
            except Exception:
                num_profile = {
                    "min": min(numeric_vals) if numeric_vals else None,
                    "max": max(numeric_vals) if numeric_vals else None,
                }

        # shape for strings
        len_stats = None
        cls_hints = None
        if str_vals:
            lens = [len(s) for s in str_vals]
            try:
                len_stats = {
                    "min_len": min(lens),
                    "max_len": max(lens),
                    "mean_len": statistics.fmean(lens),
                    "median_len": statistics.median(lens),
                }
            except Exception:
                len_stats = {
                    "min_len": min(lens),
                    "max_len": max(lens),
                }
            # aggregate simple char-class hints on a small subset
            sample_for_cls = str_vals[: min(256, len(str_vals))]
            agg = Counter()
            for s in sample_for_cls:
                cc = _char_class_profile(s)
                for k, v in cc.items():
                    agg[(k, bool(v))] += 1
            # convert to rates
            cls_hints = {}
            denom = max(1, len(sample_for_cls))
            for k in ("has_upper", "has_lower", "has_digit", "has_space", "has_punct", "all_digits", "looks_date_like", "looks_json_like"):
                cls_hints[k] = agg.get((k, True), 0) / denom

        # top-k values (from sample)
        # stringify to avoid Counter treating floats/ints differently from strings in later display
        value_counter = Counter([_strish(v) if v is not None else "<NULL>" for v in data])
        topk_values = value_counter.most_common(topk)

        # common prefixes (just from the top few non-null)
        prefixes = {}
        for s in [sv for sv in str_vals[: min(10, len(str_vals))]]:
            for k, v in _prefixes(s).items():
                prefixes.setdefault(k, Counter())
                prefixes[k][v] += 1
        top_prefixes = {k: cnt.most_common(5) for k, cnt in prefixes.items()} if prefixes else {}

        # simple min/max over raw (string) ordering to give a hint even for non-numerics
        raw_min = None
        raw_max = None
        try:
            non_null_raw = [v for v in data if v is not None]
            if non_null_raw:
                raw_min = min(non_null_raw)
                raw_max = max(non_null_raw)
        except Exception:
            pass

        # MinHash sketch (over stringified non-nulls)

        col_profiles[name] = {
            "sampled_rows": n,
            "nulls": nulls,
            "non_nulls": non_null,
            "distinct_in_sample": len(distinct_vals),
            "numeric_shape": num_profile,
            "string_len_shape": len_stats,
            # "string_charclass_hints": cls_hints,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "topk_values": topk_values,
            # "top_prefixes": top_prefixes,
        }

    return col_profiles

async def sqlite_peek(
    db_id: str,
    table: str,
    columns: List[str],
    *,
    limit: int = 10,
    timeout_s: float = 20.0,
    vm_step_limit: int = 1_000_000,
    busy_timeout_ms: int = 20_000,
    profile: bool = True,
    profile_scan_rows: int = 5000,
    profile_topk: int = 10,
    where: Optional[str] = None,
    **_extra,
) -> Dict[str, Any]:
    import time
    t0 = time.time()
    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return {"error": f"no_db_found for db_id={db_id}"}

        if not table or not isinstance(table, str):
            return {"error": "You must provide a valid table name"}
        if not columns or not isinstance(columns, list):
            return {"error": "You must provide a non-empty list of columns"}

        where_sql = ""
        if isinstance(where, str) and where.strip():
            where_sql = f" WHERE ({where.strip()})"

        col_results: Dict[str, Any] = {}

        for col in columns:
            col_q = f"`{col}`"
            tbl_q = f"`{table}`"

            # --- 1) Small preview rows ---
            sql_preview = _force_limit(f"SELECT {col_q} FROM {tbl_q}{where_sql}", limit)
            preview_rows, _ = _run_select(
                db_path,
                sql_preview,
                hard_timeout_s=float(timeout_s),
                busy_ms=int(busy_timeout_ms),
                max_steps=int(vm_step_limit),
                cap_rows=None,
            )
            preview_values = [r[0] for r in preview_rows] if preview_rows else []

            # --- 2) Larger profile sample ---
            col_profile = None
            if profile:
                sql_profile = _force_limit(f"SELECT {col_q} FROM {tbl_q}{where_sql}", profile_scan_rows)
                prof_rows, _ = _run_select(
                    db_path,
                    sql_profile,
                    hard_timeout_s=float(timeout_s),
                    busy_ms=int(busy_timeout_ms),
                    max_steps=int(vm_step_limit),
                    cap_rows=None,
                )
                flat_rows = [r[0] for r in prof_rows]
                col_profile = _profile_columns_from_sample(
                    [(v,) for v in flat_rows], [col], topk=int(profile_topk)
                )[col]

            col_results[col] = {
                "rows": preview_values,      # keep this small
                "profile": col_profile,
            }

        return {
            "db_path": db_path,
            "columns": col_results,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    except Exception as e:
        return {"error": str(e)}


async def sqlite_query(
    db_id: str,
    sql: str,
    *,
    timeout_s: float = 20.0,
    vm_step_limit: int = 5_000_000,
    busy_timeout_ms: int = 3000,
    max_return_rows: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Execute the agent's **actual** read-only SQL and return results, with guardrails.

    This function is intended as the **ultimate verification step** before producing
    a `<final_answer>` with `<sql_code>...</sql_code>`.The query may need rewriting if the results are different from expectations. In a text-to-SQL workflow,
    the LLM may use lighter probes (`sqlite_peek`, `bm25_search_sqlite`) during
    reasoning. But before committing to an answer, it should always call
    `sqlite_query` once with the **exact final SQL** it intends to surface.

    Why:
      • Ensures the SQL is executable on the target DB (valid syntax, schema alignment).
      • Confirms that results are returned within resource and safety limits.
      • Catches runtime errors (bad joins, invalid column names, etc.) before vending.
      • Allows truncation detection if the result set is too large.

    Args:
      db_id: Logical database id (e.g., "financial", "formula_1").
      sql: Final SELECT-only SQL. No automatic LIMIT is injected here;
           the agent must include LIMIT if result cardinality may be large.
      timeout_s: Wallclock guard (default 8s).
      vm_step_limit: SQLite VM step cap (default 5,000,000).
      busy_timeout_ms: SQLite busy timeout for readonly connection (default 3000).
      max_return_rows: If set, cap returned rows and set "truncated": true when reached.

    Returns:
      {"rows":[...], "columns":[...], "db_path": "...", "truncated": bool}
      or {"error":"..."}.

    Notes:
      - Enforces **SELECT-only**.
      - Executes with SQLite query_only=ON (read-only, no writes).
      - Always run this once per query before vending `<final_answer>`.
      - If a result set is massive, only the first `max_return_rows` are returned
        and `"truncated": true` is set, signaling the agent to revise with LIMIT.
    """
    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return {"error": f"no_db_found for db_id={db_id}"}
        rows, cols = _run_select(
            db_path,
            sql,
            hard_timeout_s=float(timeout_s),
            busy_ms=int(busy_timeout_ms),
            max_steps=int(vm_step_limit),
            cap_rows=int(max_return_rows) if max_return_rows is not None else None,
        )
        truncated = False
        if max_return_rows is not None and len(rows) >= max_return_rows:
            truncated = True
        return {"rows": rows, "columns": cols, "truncated": truncated}
    except Exception as e:
        return {"error": str(e)}



async def bm25_search_sqlite(
    db_id: str,
    table: str,
    column: str,
    query: str,
    top_k: int = 10,
    *,
    timeout_s: Optional[float] = None,
    busy_timeout_ms: Optional[int] = None,
    vm_step_limit: Optional[int] = None,
    where: Optional[str] = None,
    distinct: bool = True,
    case_sensitive: bool = False,
    min_chars: int = 0,
    max_chars: int = 200
) -> List[Dict[str, Any]]:
    def _tok(s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())

    try:
        if not isinstance(query, str) or not query.strip():
            return [{"error": "invalid_query"}]
        if not table or not column:
            return [{"error": "missing_table_or_column"}]
        k = max(1, int(top_k))
    except Exception as e:
        return [{"error": f"validation_error: {e}"}]

    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return [{"error": f"no_db_found for db_id={db_id}"}]
    except Exception as e:
        return [{"error": f"resolve_error: {e}"}]

    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        return [{"error": f"missing_dependency: rank-bm25 not installed ({e})"}]

    try:
        conn = _connect_ro(db_path, busy_ms=int(busy_timeout_ms) if busy_timeout_ms is not None else 2000)
        if vm_step_limit or timeout_s:
            _set_progress_guard(
                conn,
                hard_timeout_s=float(timeout_s) if timeout_s is not None else 20.0,
                max_steps=int(vm_step_limit) if vm_step_limit is not None else 0
            )

        cur = conn.cursor()
        col_q = f"`{column}`"
        tbl_q = f"`{table}`"
        distinct_sql = "DISTINCT " if distinct else ""
        where_sql = ""
        if where and isinstance(where, str) and where.strip():
            where_sql = f" WHERE ({where.strip()}) AND {col_q} IS NOT NULL AND TRIM({col_q}) != ''"
        else:
            where_sql = f" WHERE {col_q} IS NOT NULL AND TRIM({col_q}) != ''"
        collation = "" if case_sensitive else " COLLATE NOCASE"

        sql = f"SELECT {distinct_sql}{col_q} FROM {tbl_q}{where_sql}{collation};"
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        return [{"error": f"sqlite_error: {e}"}]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    values: List[str] = []
    for r in rows:
        v = r[0]
        if isinstance(v, str):
            s = v.strip()
            if s and len(s) >= int(min_chars):
                values.append(s[: int(max_chars)])

    if not values:
        return []

    corpus: List[List[str]] = []
    kept: List[str] = []
    for v in values:
        toks = _tok(v)
        if toks:
            corpus.append(toks)
            kept.append(v)

    if not corpus:
        return []

    q_tokens = _tok(query)
    if not q_tokens:
        return [{"error": "invalid_query_tokens"}]

    try:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [
            {"id": f"{table}-{column}-{idx}", "contents": kept[idx], "score": float(score)}
            for idx, score in ranked
        ]
    except Exception as e:
        return [{"error": f"bm25_error: {e}"}]
