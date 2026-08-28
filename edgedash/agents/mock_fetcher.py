"""
edgedash/agents/mock_fetcher.py

MockFetcher — returns 12 realistic fake job listings for the configured role
and city. No network calls. Used to develop and test the pipeline.

Dedup guarantee: 4 of the 12 listings have fixed, stable URLs so the same
rows will be IGNORED on the second run, proving INSERT OR IGNORE works.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from edgedash.agents.base import Agent, AgentResult
from edgedash.config import Config
from edgedash import storage
from edgedash.planning import StopConditions

_SOURCE = "mock"

# These 4 URLs are intentionally identical every run — dedup bait.
_STABLE_URLS = [
    "https://jobs.mock/stable-001",
    "https://jobs.mock/stable-002",
    "https://jobs.mock/stable-003",
    "https://jobs.mock/stable-004",
]


def _days_ago(n: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.date().isoformat()


def _build_listings(role: str, city: str) -> list[dict]:
    """
    Construct 12 fake listings: 4 stable (same every run) + 8 timestamped
    (unique per run via a date-based URL fragment).
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    stable = [
        {
            "title": f"Junior {role}",
            "company": "Infosys Analytics",
            "location": city,
            "url": _STABLE_URLS[0],
            "description": (
                f"Looking for a junior {role} to join our BI team. "
                "Must know SQL, Excel, and basic Python for data wrangling. "
                "Experience with Power BI dashboards is a plus."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(5),
        },
        {
            "title": f"Senior {role}",
            "company": "Flipkart Data",
            "location": city,
            "url": _STABLE_URLS[1],
            "description": (
                f"Senior {role} role in our supply-chain intelligence team. "
                "Strong Python (Pandas, NumPy), SQL, and Tableau required. "
                "Spark experience and dbt knowledge strongly preferred."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(3),
        },
        {
            "title": f"{role} – Growth",
            "company": "Razorpay",
            "location": city,
            "url": _STABLE_URLS[2],
            "description": (
                "Own analytics for our growth and payments funnel. "
                "We use Redshift, dbt, and Looker. You should be fluent in SQL "
                "and comfortable writing Python scripts for ETL pipelines."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(2),
        },
        {
            "title": f"Lead {role}",
            "company": "Swiggy",
            "location": city,
            "url": _STABLE_URLS[3],
            "description": (
                f"Lead a team of 3 analysts. Define KPIs, build Tableau dashboards, "
                "and present insights to product leadership. "
                "7+ years experience required. Apache Airflow and Spark a bonus."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(1),
        },
    ]

    dynamic = [
        {
            "title": f"{role} – Fintech",
            "company": "PhonePe",
            "location": city,
            "url": f"https://jobs.mock/phonepe-{role.lower().replace(' ','-')}-{today}",
            "description": (
                "Analyse transaction patterns and fraud signals. "
                "Strong SQL required; Python (scikit-learn) is a plus. "
                "You will work closely with the ML and risk teams."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(0),
        },
        {
            "title": f"Associate {role}",
            "company": "Wipro Digital",
            "location": city,
            "url": f"https://jobs.mock/wipro-associate-{today}",
            "description": (
                "Entry-level role supporting client reporting pipelines. "
                "Must know Excel and SQL. Exposure to Power BI preferred. "
                "Good communication skills essential."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(0),
        },
        {
            "title": f"{role} – Healthcare",
            "company": "Apollo Healthtech",
            "location": city,
            "url": f"https://jobs.mock/apollo-{today}",
            "description": (
                "Build dashboards for clinical and operational metrics. "
                "Python, SQL, and Tableau experience needed. "
                "Domain exposure to healthcare data a big advantage."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(1),
        },
        {
            "title": f"Contract {role}",
            "company": "Mindtree",
            "location": city,
            "url": f"https://jobs.mock/mindtree-contract-{today}",
            "description": (
                "6-month contract for a client migration project. "
                "Requires SQL, Python pandas, and strong data-cleaning skills. "
                "Immediate joiners preferred."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(2),
        },
        {
            "title": f"{role} – E-commerce",
            "company": "Meesho",
            "location": city,
            "url": f"https://jobs.mock/meesho-ecom-{today}",
            "description": (
                "Drive seller and buyer analytics on our marketplace. "
                "Deep SQL expertise required; dbt, Airflow, and Spark preferred. "
                "Python scripting for automation is expected."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(0),
        },
        {
            "title": f"Mid-Level {role}",
            "company": "CRED",
            "location": city,
            "url": f"https://jobs.mock/cred-mid-{today}",
            "description": (
                "Work on credit risk and member-behaviour analytics. "
                "Must be strong in SQL and Python. "
                "Exposure to A/B testing and experimentation frameworks preferred."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(1),
        },
        {
            "title": f"{role} Intern (PPO)",
            "company": "Ola Electric",
            "location": city,
            "url": f"https://jobs.mock/ola-intern-{today}",
            "description": (
                "6-month internship with pre-placement offer for strong performers. "
                "Basic Python and SQL required. "
                "You will build dashboards in Tableau and automate weekly reports."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(0),
        },
        {
            "title": f"Principal {role}",
            "company": "Walmart Global Tech",
            "location": city,
            "url": f"https://jobs.mock/walmart-principal-{today}",
            "description": (
                "Define the analytics strategy for India supply-chain operations. "
                "10+ years required. Must be expert in SQL, Python, Spark, and dbt. "
                "Experience presenting to VP-level stakeholders essential."
            ),
            "source": _SOURCE,
            "posted_at": _days_ago(3),
        },
    ]

    return stable + dynamic


class MockFetcher:
    name: str = "MockFetcher"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        listings = _build_listings(config.target_role, config.target_city)
        new_count = storage.upsert_listings(db_path, listings)

        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=new_count,
            notes=(
                f"Generated {len(listings)} listings; "
                f"{new_count} were new, "
                f"{len(listings) - new_count} were duplicates (deduped)."
            ),
        )
