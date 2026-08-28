"""
edgedash/skills.py

Deterministic skill name canonicalisation. No network. No model. (Rule 22-23)

Public API
----------
    canonical(raw: str, aliases: dict) -> str

CLI
---
    python -m edgedash.skills --audit
        Reads every extracted required_skill from the DB and prints:
          - Top 40 raw strings with counts and their canonical forms
          - Singletons (seen exactly once) — the junk drawer

    python -m edgedash.skills --suggest-aliases
        ONE model call proposing groupings of unaliased skill strings.
        Prints ready-to-paste YAML. Writes nothing. You decide what to add.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# canonical() — show this to the user first
# ---------------------------------------------------------------------------

# Parenthetical qualifiers to strip: "kubernetes (eks)" -> "kubernetes"
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*")
# Collapse runs of whitespace/punctuation after stripping
_SPACE_RE = re.compile(r"\s+")
# Leading/trailing punctuation to strip (but keep internal hyphens in "ci/cd")
_EDGE_PUNCT_RE = re.compile(r"^[^\w#/\-\.]+|[^\w#/\-\.]+$")


def canonical(raw: str, aliases: dict[str, list[str]]) -> str:
    """Return the canonical form of *raw* skill string.

    Steps (applied in order):
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove parenthetical qualifiers  e.g. "Kubernetes (EKS)" -> "kubernetes"
      4. Strip leading/trailing punctuation
      5. Collapse internal whitespace runs to single space
      6. Look up in alias map; return canonical name if found, else the cleaned string

    Pure function — same input always produces same output.
    """
    if not raw:
        return ""

    s = raw.lower()
    s = s.strip()
    s = _PAREN_RE.sub(" ", s)       # drop parentheticals
    s = _EDGE_PUNCT_RE.sub("", s)   # strip edge punctuation
    s = _SPACE_RE.sub(" ", s)       # collapse whitespace
    s = s.strip()

    # Build reverse lookup: alias -> canonical
    # (build inline so this stays a pure function with no module-level state)
    lookup = _build_lookup(aliases)
    return lookup.get(s, s)


def _build_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Return alias->canonical dict from the config alias map."""
    lookup: dict[str, str] = {}
    for canon, alias_list in aliases.items():
        canon_clean = canon.lower().strip()
        lookup[canon_clean] = canon_clean
        for alias in (alias_list or []):
            lookup[alias.lower().strip()] = canon_clean
    return lookup


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _load_all_raw_skills(db_path: str) -> list[str]:
    """Read every required_skill string from extraction_cache (read-only)."""
    import json, sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT extraction_json FROM extraction_cache").fetchall()
    conn.close()
    skills: list[str] = []
    for (raw_json,) in rows:
        try:
            data = json.loads(raw_json)
            skills.extend(data.get("required_skills") or [])
        except (json.JSONDecodeError, AttributeError):
            continue
    return skills


def _run_audit(db_path: str, aliases: dict[str, list[str]]) -> None:
    raw_skills = _load_all_raw_skills(db_path)
    if not raw_skills:
        print("No extracted skills found. Run the Scorer first.")
        return

    counts: Counter[str] = Counter(raw_skills)
    total = sum(counts.values())
    print(f"\nTotal raw skill occurrences across all cached extractions: {total}")
    print(f"Unique raw strings: {len(counts)}\n")

    # --- Top 40 ---
    print("=" * 72)
    print(f"{'#':<4} {'RAW STRING':<35} {'COUNT':>5}  {'CANONICAL FORM'}")
    print("=" * 72)
    for rank, (raw, count) in enumerate(counts.most_common(40), 1):
        canon = canonical(raw, aliases)
        marker = "  *" if canon != raw else ""
        print(f"{rank:<4} {raw:<35} {count:>5}  {canon}{marker}")
    print("  (* = alias applied)")

    # --- Singletons ---
    singletons = sorted(s for s, c in counts.items() if c == 1)
    print(f"\n{'=' * 72}")
    print(f"SINGLETONS ({len(singletons)}) — seen exactly once.")
    print("These are typos, junk phrases, or full sentences the extractor")
    print("mistakenly captured. Review to find aliases you're missing.")
    print("=" * 72)
    for s in singletons:
        print(f"  {s}")


# ---------------------------------------------------------------------------
# Alias suggestion  (one LLM call, read-only, rule 23)
# ---------------------------------------------------------------------------

_SUGGEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["suggestions"],
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["canonical", "variants", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "canonical": {"type": "string"},
                    "variants":  {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "confidence": {"type": "string", "enum": ["high", "low"]},
                },
            },
        }
    },
}

_SUGGEST_PROMPT_TEMPLATE = """\
You are a technical recruiter with deep knowledge of software and data engineering skills.

Below is a list of skill strings extracted from real job listings, with their occurrence counts.
These strings have already been lowercased and cleaned.
They are NOT yet in any alias map — they are the leftovers after known aliases were applied.

Your task: identify groups of strings that refer to the SAME underlying technical skill.
For each group, choose ONE canonical name (the clearest, most common form) and list the others as variants.

IMPORTANT RULES:
- Only group strings that are genuinely the same skill. When unsure, do NOT group them.
- Node.js and JavaScript are DIFFERENT skills. Do not merge them.
- A framework and its host language are DIFFERENT skills (e.g. React and JavaScript).
- Prefer "high" confidence only when the grouping is unambiguous (abbreviation, typo, plural form).
- Use "low" confidence for anything arguable.
- Do not invent new strings. Only use strings from the list below.
- Reply with a JSON object only. No markdown fences. No prose.

SKILL STRINGS (format: "skill string" — N occurrences):
{skill_list}

Return JSON matching this schema:
{{
  "suggestions": [
    {{"canonical": "the preferred name", "variants": ["alt1", "alt2"], "confidence": "high"}}
  ]
}}
If you find no groupings worth suggesting, return {{"suggestions": []}}."""


def _collect_unaliased_canonicals(
    db_path: str,
    aliases: dict[str, list[str]],
    min_count: int = 2,
) -> Counter[str]:
    """Return canonical forms of skills that are NOT already in the alias map,
    with counts >= min_count.  Read-only."""
    raw_skills = _load_all_raw_skills(db_path)
    counts: Counter[str] = Counter(raw_skills)

    # Build the full set of strings already covered by the alias map
    covered: set[str] = set()
    for canon, alias_list in aliases.items():
        covered.add(canon.lower().strip())
        for a in (alias_list or []):
            covered.add(a.lower().strip())

    # Canonicalise raw counts and filter out covered strings
    canon_counts: Counter[str] = Counter()
    for raw, count in counts.items():
        c = canonical(raw, aliases)
        if c not in covered:
            canon_counts[c] += count

    # Only pass strings with meaningful frequency — singletons are noise
    return Counter({s: n for s, n in canon_counts.items() if n >= min_count})


def _detect_conflicts(
    suggestions: list[dict[str, Any]],
    aliases: dict[str, list[str]],
) -> list[str]:
    """Return conflict messages for proposals that would merge separate alias-map entries."""
    # Existing canonical keys in the alias map (the ones the user explicitly separated)
    existing_canonicals: set[str] = {k.lower().strip() for k in aliases}

    conflicts: list[str] = []
    for s in suggestions:
        canon = s["canonical"].lower().strip()
        variants = [v.lower().strip() for v in s.get("variants", [])]
        all_terms = {canon} | set(variants)
        # Find terms that are existing canonical keys
        clashing = all_terms & existing_canonicals
        if len(clashing) > 1:
            conflicts.append(
                f"  CONFLICT: proposal wants to merge {sorted(clashing)} — "
                f"but these are SEPARATE entries in your alias map."
            )
        elif len(clashing) == 1:
            # Canonical of the proposal matches one alias-map entry; variants might clash another
            # Check if any variant is also a canonical key different from the matching one
            matched = clashing.pop()
            other_clashing = (all_terms - {matched}) & existing_canonicals
            if other_clashing:
                conflicts.append(
                    f"  CONFLICT: proposal merges '{matched}' with "
                    f"{sorted(other_clashing)} — these are separate alias-map entries."
                )
    return conflicts


def _format_yaml_block(s: dict[str, Any]) -> str:
    """Format one suggestion as a ready-to-paste config.yaml alias-map entry."""
    canon = s["canonical"]
    variants = s["variants"]
    conf = s["confidence"]
    lines = [f'  "{canon}":  # confidence: {conf}']
    for v in variants:
        lines.append(f'    - "{v}"')
    return "\n".join(lines)


def _run_suggest_aliases(db_path: str, aliases: dict[str, list[str]]) -> None:
    # ── Collect unaliased skill strings ──────────────────────────────────
    unaliased = _collect_unaliased_canonicals(db_path, aliases, min_count=2)
    if not unaliased:
        print("\nNo unaliased skill strings with count >= 2 found.")
        print("Your alias map already covers everything — or run more scoring cycles first.")
        return

    skill_list = "\n".join(
        f'"{skill}" — {count} occurrences'
        for skill, count in unaliased.most_common(120)   # cap prompt size
    )

    print(f"\nCollected {len(unaliased)} unaliased skill strings. Sending to model…\n")

    # ── One LLM call (rule 23: model may SUGGEST only) ────────────────────
    from edgedash.llm import complete_json, LLMError
    from edgedash.config import load_config
    cfg = load_config()

    prompt = _SUGGEST_PROMPT_TEMPLATE.format(skill_list=skill_list)
    try:
        result = complete_json(prompt, _SUGGEST_SCHEMA, max_retries=1, _cfg=cfg)
    except LLMError as exc:
        print(f"Model call failed: {exc}")
        return

    suggestions: list[dict[str, Any]] = result.get("suggestions", [])
    if not suggestions:
        print("Model found no groupings worth suggesting. Your skills look well-separated.")
        return

    # ── Conflict detection ────────────────────────────────────────────────
    conflicts = _detect_conflicts(suggestions, aliases)

    # ── Print output — read-only, nothing written ─────────────────────────
    print("!" * 72)
    print("  ALIAS SUGGESTIONS — REVIEW REQUIRED BEFORE USE")
    print()
    print("  These are MODEL PROPOSALS, not decisions.")
    print("  Over-merging is worse than under-merging.")
    print("  Merging distinct skills produces a report that looks clean")
    print("  but tells you the wrong thing. When unsure, leave separate.")
    print("!" * 72)

    if conflicts:
        print()
        print("  ⚠  CONFLICTS WITH YOUR EXISTING ALIAS MAP:")
        for c in conflicts:
            print(c)
        print("  These proposals contradict choices you already made.")
        print("  They are shown below but should be treated with extra scepticism.")

    high = [s for s in suggestions if s.get("confidence") == "high"]
    low  = [s for s in suggestions if s.get("confidence") != "high"]

    sep = "─" * 72

    if high:
        print(f"\n{sep}")
        print(f"  HIGH CONFIDENCE ({len(high)} groups) — paste into skill_aliases in config.yaml")
        print(sep)
        for s in high:
            print(_format_yaml_block(s))
            print()

    if low:
        print(f"\n{sep}")
        print(f"  LOW CONFIDENCE ({len(low)} groups) — review carefully before adding")
        print(sep)
        for s in low:
            print(_format_yaml_block(s))
            print()

    print(sep)
    print(f"  {len(suggestions)} proposals · {len(high)} high · {len(low)} low · "
          f"{len(conflicts)} conflict(s) with existing map")
    print("  NO FILES WERE MODIFIED. Add entries to config.yaml manually.")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeDash skill canonicalisation tools")
    parser.add_argument("--audit", action="store_true",
                        help="Print skill frequency audit from the database")
    parser.add_argument("--db", default="edgedash.db",
                        help="Path to the SQLite database (default: edgedash.db)")
    args = parser.parse_args()

    if args.audit:
        from edgedash.config import load_config
        cfg = load_config()
        _run_audit(args.db, cfg.skill_aliases)
    else:
        parser.print_help()
