"""
edgedash/query/ask.py

Two-call query pipeline: route → execute → phrase.

Rules honoured
--------------
  Rule 42: route with one LLM call that sees ONLY tool names + descriptions.
  Rule 43: phrase with one LLM call that sees ONLY the question + rows.
           Prompt explicitly forbids estimating or adding outside context.
  Rule 44: Answer carries both .text and .rows so the UI can show both.
  Rule 45: null tool is valid — return a fixed message, no phrasing call.

Public API
----------
    ask(question: str) -> Answer
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from edgedash import storage
from edgedash.config import load_config
from edgedash.llm import LLMError, complete_json
from edgedash.query.tools import TOOLS, ParamSpec, call

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    text: str                       # 2-3 sentence phrased response
    rows: list[dict]                # raw data rows from the tool
    tool_used: Optional[str]        # tool name, or None
    params: dict[str, Any]          # validated params passed to the tool
    summary: str                    # tool's own summary string
    confidence: str                 # "high" | "low" | "no_match"


# ---------------------------------------------------------------------------
# JSON schemas for the two LLM calls
# ---------------------------------------------------------------------------

_ROUTE_SCHEMA: dict = {
    "type": "object",
    "required": ["tool", "params", "confidence"],
    "additionalProperties": False,
    "properties": {
        "tool":       {"type": ["string", "null"]},
        "params":     {"type": "object"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
    },
}

_PHRASE_SCHEMA: dict = {
    "type": "object",
    "required": ["answer"],
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Routing prompt builder
# ---------------------------------------------------------------------------

def _build_tool_block() -> str:
    """Format the registry as a human-readable block for the router.

    Shows only names, descriptions, and parameter names with types and
    defaults. No SQL, no table names, no schema details.
    """
    lines: list[str] = []
    for name, spec in TOOLS.items():
        lines.append(f"{name}")
        # Indent description
        for desc_line in spec.description.strip().splitlines():
            lines.append(f"    {desc_line.strip()}")
        if spec.params:
            param_parts = []
            for p_name, p_spec in spec.params.items():
                bounds = ""
                if p_spec.type == "int" and p_spec.min is not None:
                    bounds = f", range {p_spec.min}–{p_spec.max}"
                param_parts.append(
                    f"{p_name}: {p_spec.type} = {p_spec.default}{bounds}"
                )
            lines.append(f"    [params: {', '.join(param_parts)}]")
        lines.append("")
    return "\n".join(lines)


_TOOL_BLOCK_CACHE: Optional[str] = None


def _tool_block() -> str:
    global _TOOL_BLOCK_CACHE
    if _TOOL_BLOCK_CACHE is None:
        _TOOL_BLOCK_CACHE = _build_tool_block()
    return _TOOL_BLOCK_CACHE


# ---------------------------------------------------------------------------
# ROUTING PROMPT  — this is the text the user asked to see first
# ---------------------------------------------------------------------------

def _routing_prompt(question: str) -> str:
    return f"""You are a routing assistant for a career intelligence system.
Your only job is to decide which tool to call to answer the user's question.

AVAILABLE TOOLS:
───────────────
{_tool_block()}

RULES (read carefully):
1. Return the name of ONE tool from the list above that best answers the question.
2. If no tool is a strong match, return null for "tool". Do NOT pick the
   closest tool and hope for the best — an explicit null is better than a
   wrong tool call. When in doubt, return null.
3. Fill "params" only with values the question explicitly states.
   Use each parameter's default for anything not mentioned.
4. "confidence" is "high" when the tool clearly matches, "low" when you
   are unsure but believe it is the right tool.
5. Reply with JSON only. No prose. No markdown fences.

QUESTION: {question}

Reply with exactly this shape and nothing else:
{{"tool": "<tool_name or null>", "params": {{}}, "confidence": "high"}}"""


# ---------------------------------------------------------------------------
# PHRASING PROMPT  — one call, rows-only, no outside context
# ---------------------------------------------------------------------------

def _phrasing_prompt(question: str, rows: list[dict], summary: str) -> str:
    rows_text = json.dumps(rows[:20], indent=2)   # cap to 20 rows in context
    return f"""You are answering a career intelligence question using only the data below.

QUESTION: {question}

DATA SUMMARY: {summary}

DATA ROWS (JSON):
{rows_text}

RULES you must follow:
- Use only numbers and facts present in these rows. Do not estimate, infer,
  or add any context from outside this data.
- Write 2 to 3 sentences only. Be specific: name companies, skills, or scores
  directly from the rows.
- If the rows are empty, say exactly: "The data does not contain an answer to
  that question."
- Do not mention tool names, SQL, or technical implementation details.
- Reply with JSON only. No prose outside the JSON. No markdown fences.

Reply with exactly this shape:
{{"answer": "<your 2-3 sentence answer here>"}}"""


# ---------------------------------------------------------------------------
# Fixed no-match response (rule 45 — no phrasing LLM call)
# ---------------------------------------------------------------------------

def _no_match_text() -> str:
    tool_descriptions = "\n".join(
        f"  • {name}: {spec.description.splitlines()[0].strip()}"
        for name, spec in TOOLS.items()
    )
    return (
        "I couldn't match your question to any available query.\n\n"
        "Here's what I can answer:\n"
        f"{tool_descriptions}\n\n"
        "Try rephrasing your question to match one of those topics."
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def ask(question: str) -> Answer:
    """Route, execute, and phrase a natural-language question.

    Two LLM calls maximum:
      1. Route: which tool + params?
      2. Phrase: convert rows to prose.

    If tool is null, returns a fixed message with no phrasing call.
    Logs every question to query_log regardless of outcome.
    """
    cfg = load_config()
    db  = cfg.db_path

    # Ensure the query_log table exists
    storage.init_db(db)

    t_start = _now_ms()

    # ── CALL 1: ROUTE ─────────────────────────────────────────────────────
    route_prompt = _routing_prompt(question)
    try:
        route_resp = complete_json(route_prompt, _ROUTE_SCHEMA, max_retries=1)
    except LLMError as exc:
        logger.error("Router LLM call failed: %s", exc)
        _write_log(db, question, None, {}, False, _now_ms() - t_start)
        return Answer(
            text=f"Routing failed: {exc}",
            rows=[],
            tool_used=None,
            params={},
            summary="",
            confidence="no_match",
        )

    tool_name  = route_resp.get("tool")
    raw_params = route_resp.get("params") or {}
    confidence = route_resp.get("confidence", "low")

    # ── Validate tool name against registry (never getattr on raw string) ──
    if tool_name is not None and tool_name not in TOOLS:
        logger.warning("Router returned unknown tool '%s' — treating as null", tool_name)
        tool_name = None

    # ── NULL tool: fixed message, no second LLM call (rule 45) ────────────
    if tool_name is None:
        _write_log(db, question, None, {}, False, _now_ms() - t_start)
        return Answer(
            text=_no_match_text(),
            rows=[],
            tool_used=None,
            params={},
            summary="",
            confidence="no_match",
        )

    # ── EXECUTE the tool with validated, clamped params ───────────────────
    try:
        tool_result = call(tool_name, **raw_params)
    except Exception as exc:
        logger.error("Tool '%s' raised: %s", tool_name, exc)
        _write_log(db, question, tool_name, raw_params, False, _now_ms() - t_start)
        return Answer(
            text=f"The query ran into a problem: {exc}",
            rows=[],
            tool_used=tool_name,
            params=raw_params,
            summary="",
            confidence=confidence,
        )

    rows:    list[dict] = tool_result.get("rows", [])
    summary: str        = tool_result.get("summary", "")

    # ── CALL 2: PHRASE ────────────────────────────────────────────────────
    phrase_prompt = _phrasing_prompt(question, rows, summary)
    try:
        phrase_resp = complete_json(phrase_prompt, _PHRASE_SCHEMA, max_retries=1)
        answer_text = phrase_resp.get("answer", "")
    except LLMError as exc:
        logger.warning("Phrasing LLM call failed: %s — using summary fallback", exc)
        answer_text = summary   # fall back to the tool's own summary

    answerable = bool(rows)
    _write_log(db, question, tool_name, raw_params, answerable, _now_ms() - t_start)

    return Answer(
        text=answer_text,
        rows=rows,
        tool_used=tool_name,
        params=raw_params,
        summary=summary,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _write_log(
    db: str,
    question: str,
    tool_chosen: Optional[str],
    params: dict,
    answerable: bool,
    duration_ms: int,
) -> None:
    try:
        storage.log_query(
            path=db,
            question=question,
            tool_chosen=tool_chosen,
            params_json=json.dumps(params),
            answerable=answerable,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        logger.warning("Could not write to query_log: %s", exc)
