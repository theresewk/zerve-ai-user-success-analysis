
import pandas as pd
import numpy as np

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Load the CSV file
print("Loading CSV file...")
event_df = pd.read_csv("app/zerve_hackathon_for_reviewc8fa7c7.csv", low_memory=False)

print(f"\n{'='*60}")
print(f"  DATASET OVERVIEW")
print(f"{'='*60}")
print(f"  Rows    : {len(event_df):,}")
print(f"  Columns : {event_df.shape[1]}")

print(f"\n{'='*60}")
print(f"  SCHEMA: COLUMN DTYPES & NULL AUDIT")
print(f"{'='*60}")

null_summary = pd.DataFrame({
    "dtype":      event_df.dtypes.astype(str),
    "non_null":   event_df.notna().sum(),
    "null_count": event_df.isna().sum(),
    "null_rate_%": (event_df.isna().mean() * 100).round(2),
    "unique_values": event_df.nunique()
})
print(null_summary.to_string())

# ── Key columns for unique value samples ──────────────────────────────────────
key_cols_samples = {
    "event":           10,
    "os":              15,
    "browser":         15,
    "device_type":     15,
    "geoip_country":   15,
    "referring_domain": 10,
    "surface":         10,
    "python_version":  10,
    "python_runtime":  10,
    "tool_name":       10,
}

print(f"\n{'='*60}")
print(f"  UNIQUE VALUE SAMPLES FOR KEY COLUMNS")
print(f"{'='*60}")
for col, n in key_cols_samples.items():
    matches = [c for c in event_df.columns if col in c.lower()]
    # prefer direct/short column name, avoid prop_$set / prop_$set_once
    matches = [m for m in matches if "$set" not in m] or matches
    for match_col in matches[:1]:
        top_vals = event_df[match_col].value_counts().head(n)
        print(f"\n  [{match_col}] — top {n} of {event_df[match_col].nunique():,} unique:")
        _samples_df = top_vals.reset_index()
        _samples_df.columns = ["value", "count"]
        _samples_df["pct_%"] = (_samples_df["count"] / len(event_df) * 100).round(2)
        print(_samples_df.to_string(index=False))

# ── Top 30 events ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TOP 30 MOST FREQUENT EVENT NAMES")
print(f"{'='*60}")

event_col = "event"   # confirmed clean column
top_events = event_df[event_col].value_counts().head(30)
print(f"\n  Column: '{event_col}'  |  Total unique events: {event_df[event_col].nunique():,}\n")

_top_events_df = top_events.reset_index()
_top_events_df.columns = ["event_name", "count"]
_top_events_df["pct_%"] = (_top_events_df["count"] / len(event_df) * 100).round(2)
_top_events_df.insert(0, "rank", range(1, len(_top_events_df) + 1))
print(_top_events_df.to_string(index=False))

print(f"\n{'='*60}")
print(f"  SCHEMA PROFILE COMPLETE")
print(f"{'='*60}")
