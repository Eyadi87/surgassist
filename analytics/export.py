import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlite3
import pandas as pd
from pathlib import Path
from analytics.analyzer import UsageAnalyzer


def export_full_report(db_path=None, out_dir=None, config_path=None):
    if db_path     is None: db_path     = os.path.join(ROOT, "data", "surgassist.db")
    if out_dir     is None: out_dir     = os.path.join(ROOT, "data", "exports")
    if config_path is None: config_path = os.path.join(ROOT, "config.yaml")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM instrument_events", conn)
    conn.close()

    if df.empty:
        print("No data to export. Run the dashboard and process a video first.")
        return None

    analyzer  = UsageAnalyzer(config_path)
    stats     = analyzer.compute_stats(df)
    switches  = analyzer.compute_switch_sequence(df)
    waste_pct = analyzer.waste_percentage(stats)
    or_score  = analyzer.or_efficiency_score(stats)

    df.to_csv(f"{out_dir}/raw_events.csv", index=False)
    print(f"✅ raw_events.csv          — {len(df)} rows")

    rows = []
    for inst, s in stats.items():
        rows.append({
            "instrument":    inst,
            "usage_sec":     s["usage_sec"],
            "total_dets":    s["total_frames"],
            "first_seen":    s["first_seen"],
            "idle_ratio":    s["idle_ratio"],
            "waste_flag":    s["is_waste"],
        })
    pd.DataFrame(rows).to_csv(f"{out_dir}/usage_summary.csv", index=False)
    print(f"✅ usage_summary.csv       — {len(rows)} instruments")

    pd.DataFrame(switches, columns=["from", "to"]).to_csv(
        f"{out_dir}/switch_log.csv", index=False
    )
    print(f"✅ switch_log.csv          — {len(switches)} switches")

    pd.DataFrame([{
        "waste_pct":      waste_pct,
        "or_efficiency":  or_score,
        "total_switches": len(switches),
        "max_frame":      int(df["frame_id"].max()),
        "unique_instruments": int(df["label"].nunique()),
    }]).to_csv(f"{out_dir}/metrics_summary.csv", index=False)
    print(f"✅ metrics_summary.csv")
    print(f"\nAll exports saved to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    import glob
    # Find DB wherever it is
    candidates = [
        os.path.join(ROOT, "data", "surgassist.db"),
        os.path.join(os.getcwd(), "data", "surgassist.db"),
    ]
    found = next((p for p in candidates if os.path.exists(p)), None)
    if not found:
        print("❌ surgassist.db not found. Run the dashboard and process a video first.")
    else:
        print(f"Using DB: {found}")
        export_full_report(db_path=found)