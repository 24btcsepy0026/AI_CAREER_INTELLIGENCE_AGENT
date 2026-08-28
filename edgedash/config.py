"""
edgedash/config.py

Loads project configuration from config.yaml at the repo root.
All user-specific values (role, city, skills, etc.) live here — never in code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# PyYAML is the only sensible standard for YAML parsing; stdlib has no YAML support.
try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required: pip install pyyaml"
    ) from exc

# Load .env into os.environ if python-dotenv is available.
# This is the one place env vars are loaded — nowhere else.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv is optional; env vars can be set by other means

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_DEFAULTS: dict = {
    "target_role": "Software Engineer",
    "target_city": "Remote",
    "keywords": [],
    "my_skills": [],
    "experience_years": 0,
    "db_path": "edgedash.db",
    "min_fit_score": 50,
    "sources": ["arbeitnow"],
    "use_mock_fetcher": False,
    "llm_provider": "gemini",
    "llm_model": "gemini-flash-lite-latest",
    # Scoring
    "target_seniority": "mid",
    "score_batch_size": 25,
    "weight_skill_match": 0.45,
    "weight_seniority_fit": 0.25,
    "weight_location_fit": 0.15,
    "weight_recency": 0.15,
    # Gap analysis
    "skill_aliases": {},          # canonical name -> list of aliases (rule 23)
    # Orchestration thresholds
    "fetch_interval_hours": 6,
    "score_max_seconds": 300,
    "analyse_max_seconds": 120,
    "fetch_max_pages": 5,
    "fetch_max_listings": 500,
    # Verification thresholds (edgedash/verification.py)
    "min_score_spread": 10,           # catches score inflation (all scores compressed)
    "min_score_stdev": 5.0,           # catches score compression via standard deviation
    "max_empty_extraction_pct": 20.0, # catches broken extractor (too many empty skill lists)
    "max_skills_per_listing": 40,     # catches extractor returning sentences as skills
    "min_gap_sample": 3,              # catches ranking a gap seen in only 1-2 listings
    "max_data_age_days": 3,           # catches stale data (no fetch for too long)
}


@dataclass
class Config:
    target_role: str
    target_city: str
    keywords: list[str]
    my_skills: list[str]
    experience_years: int
    db_path: str
    min_fit_score: int
    sources: list[str]
    use_mock_fetcher: bool
    llm_provider: str
    llm_model: str
    # Scoring
    target_seniority: str
    score_batch_size: int
    weight_skill_match: float
    weight_seniority_fit: float
    weight_location_fit: float
    weight_recency: float
    # Gap analysis
    skill_aliases: dict[str, list[str]]   # canonical -> [alias, alias, ...]
    # Orchestration thresholds
    fetch_interval_hours: int
    score_max_seconds: int
    analyse_max_seconds: int
    fetch_max_pages: int
    fetch_max_listings: int
    # Verification thresholds
    min_score_spread: int
    min_score_stdev: float
    max_empty_extraction_pct: float
    max_skills_per_listing: int
    min_gap_sample: int
    max_data_age_days: int


def load_config(path: Path = _CONFIG_PATH) -> Config:
    """Load and validate Config from a YAML file.

    Raises FileNotFoundError with a clear message if the file is absent.
    Missing individual fields fall back to sensible defaults.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}.\n"
            "Create one at the repo root. See config.yaml.example for the shape."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    merged = {**_DEFAULTS, **raw}

    # Validate types for fields that must be specific types.
    if not isinstance(merged["keywords"], list):
        raise ValueError("config.yaml: 'keywords' must be a list.")
    if not isinstance(merged["my_skills"], list):
        raise ValueError("config.yaml: 'my_skills' must be a list.")
    if not isinstance(merged["experience_years"], int):
        raise ValueError("config.yaml: 'experience_years' must be an integer.")
    if not isinstance(merged["min_fit_score"], int):
        raise ValueError("config.yaml: 'min_fit_score' must be an integer.")
    if not isinstance(merged["sources"], list):
        raise ValueError("config.yaml: 'sources' must be a list.")
    if not isinstance(merged["use_mock_fetcher"], bool):
        raise ValueError("config.yaml: 'use_mock_fetcher' must be a boolean.")

    return Config(
        target_role=str(merged["target_role"]),
        target_city=str(merged["target_city"]),
        keywords=list(merged["keywords"]),
        my_skills=list(merged["my_skills"]),
        experience_years=int(merged["experience_years"]),
        db_path=str(merged["db_path"]),
        min_fit_score=int(merged["min_fit_score"]),
        sources=list(merged["sources"]),
        use_mock_fetcher=bool(merged["use_mock_fetcher"]),
        llm_provider=str(merged["llm_provider"]),
        llm_model=str(merged["llm_model"]),
        target_seniority=str(merged["target_seniority"]),
        score_batch_size=int(merged["score_batch_size"]),
        weight_skill_match=float(merged["weight_skill_match"]),
        weight_seniority_fit=float(merged["weight_seniority_fit"]),
        weight_location_fit=float(merged["weight_location_fit"]),
        weight_recency=float(merged["weight_recency"]),
        skill_aliases=dict(merged["skill_aliases"]),
        fetch_interval_hours=int(merged["fetch_interval_hours"]),
        score_max_seconds=int(merged["score_max_seconds"]),
        analyse_max_seconds=int(merged["analyse_max_seconds"]),
        fetch_max_pages=int(merged["fetch_max_pages"]),
        fetch_max_listings=int(merged["fetch_max_listings"]),
        min_score_spread=int(merged["min_score_spread"]),
        min_score_stdev=float(merged["min_score_stdev"]),
        max_empty_extraction_pct=float(merged["max_empty_extraction_pct"]),
        max_skills_per_listing=int(merged["max_skills_per_listing"]),
        min_gap_sample=int(merged["min_gap_sample"]),
        max_data_age_days=int(merged["max_data_age_days"]),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
