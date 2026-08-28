import sqlite3, json
conn = sqlite3.connect('edgedash.db')

# table columns
for tbl in ['listings', 'gap_snapshots_v2', 'extraction_cache', 'cycle_log']:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    print(f"{tbl}: {cols}")

# sample a listing row (non-description fields)
row = conn.execute(
    "SELECT id, title, company, location, source, posted_at, fetched_at, "
    "fit_score, fit_reason FROM listings WHERE fit_score IS NOT NULL "
    "ORDER BY fit_score DESC LIMIT 1"
).fetchone()
print("\nbest listing sample:", dict(zip(
    ['id','title','company','location','source','posted_at','fetched_at','fit_score','fit_reason'], row
)))

# sample gap_snapshots_v2
row2 = conn.execute(
    "SELECT * FROM gap_snapshots_v2 ORDER BY opportunity_cost DESC LIMIT 1"
).fetchone()
if row2:
    cols2 = [r[1] for r in conn.execute("PRAGMA table_info(gap_snapshots_v2)").fetchall()]
    print("\nbest gap sample:", dict(zip(cols2, row2)))

# distinct run_ats
run_ats = conn.execute(
    "SELECT DISTINCT run_at FROM gap_snapshots_v2 ORDER BY run_at DESC LIMIT 5"
).fetchall()
print("\nlatest gap run_ats:", [r[0][:19] for r in run_ats])

# skill demand: sample extraction_cache
row3 = conn.execute("SELECT extraction_json FROM extraction_cache LIMIT 1").fetchone()
if row3:
    d = json.loads(row3[0])
    print("\nextraction_cache sample keys:", list(d.keys()))
    print("required_skills sample:", d.get('required_skills', [])[:5])
    print("nice_to_have sample:", d.get('nice_to_have', [])[:5])

conn.close()
