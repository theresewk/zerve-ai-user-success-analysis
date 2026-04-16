
import pandas as pd
import numpy as np
import os
import io
import sys

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Constants ─────────────────────────────────────────────────────────────────
LOW_N = 30
FLAGS = [
    "run_within_24h",
    "run_within_7d",
    "run_within_first_session",
    "runs_on_2plus_distinct_days_within_14d",
    "runs_in_3plus_distinct_sessions_within_14d",
    "returned_after_day7",
    "shipped",
    "ship_then_use",
]
TIME_COLS = [
    "time_to_first_canvas",
    "time_to_first_build",
    "time_to_first_run",
    "time_to_first_ship",
]

# ── Helper: per-user modal value from event_df ─────────────────────────────────
def user_modal(col_name):
    """Return Series[person_id → modal non-null value of col_name]."""
    _tmp = event_df[["person_id", col_name]].dropna(subset=[col_name])
    return (
        _tmp.groupby("person_id")[col_name]
        .agg(lambda x: x.mode().iloc[0])
    )

# ── Build per-user segment profile ────────────────────────────────────────────
print("Building per-user segment profiles...")
seg_cols_map = {
    "os":               "prop_$os",
    "browser":          "prop_$browser",
    "device_type":      "prop_$device_type",
    "country":          "prop_$geoip_country_name",
    "referring_domain": "prop_$session_entry_referring_domain",
    "surface":          "prop_surface",
}

seg_frames = {}
for seg_name, raw_col in seg_cols_map.items():
    seg_frames[seg_name] = user_modal(raw_col).rename(seg_name)

seg_profile = pd.concat(seg_frames.values(), axis=1).reset_index()

# Top 10 referring domains
top10_referrers = (
    seg_profile["referring_domain"].value_counts().head(10).index.tolist()
)
seg_profile["referrer_top10"] = seg_profile["referring_domain"].where(
    seg_profile["referring_domain"].isin(top10_referrers), other="Other"
)

# Product surface from pathname (modal per user)
_path_tmp = event_df[["person_id", "prop_$pathname"]].dropna(subset=["prop_$pathname"])
_path_modal = (
    _path_tmp.groupby("person_id")["prop_$pathname"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("entry_path")
)

def derive_surface(path):
    if pd.isna(path):
        return "unknown"
    p = str(path).lower().strip("/")
    if p == "" or p == "/":
        return "home"
    root = p.split("/")[0]
    mapping = {
        "canvas": "canvas", "canvases": "canvas",
        "apps": "apps", "app": "apps",
        "api": "api", "apis": "api",
        "settings": "settings", "profile": "settings",
        "docs": "docs", "documentation": "docs",
        "login": "auth", "signup": "auth", "register": "auth",
        "pricing": "pricing", "billing": "pricing",
        "dashboard": "dashboard",
    }
    return mapping.get(root, root[:20])

seg_profile = seg_profile.merge(_path_modal.reset_index(), on="person_id", how="left")
seg_profile["product_surface"] = seg_profile["entry_path"].apply(derive_surface)

# Merge segments into ladder
ladder = user_ladder_df.merge(seg_profile, on="person_id", how="left")
print(f"  ladder enriched → {ladder.shape}")
print(f"  Top 10 referrers: {top10_referrers}\n")


# ── Helpers: build rate/time tables as DataFrames ─────────────────────────────
def _build_rate_table_df(df, segment_col=None, show_aggregate=False):
    """Build a DataFrame of flag rates."""
    short_flags = [
        f.replace("run_within_24h", "run_24h")
         .replace("run_within_7d", "run_7d")
         .replace("run_within_first_session", "run_1sess")
         .replace("runs_on_2plus_distinct_days_within_14d", "2d_14d")
         .replace("runs_in_3plus_distinct_sessions_within_14d", "3s_14d")
         .replace("returned_after_day7", "ret_d7")
         .replace("ship_then_use", "ship_use")
         .replace("shipped", "shpd")
        for f in FLAGS
    ]

    if segment_col is None:
        rows = []
        n_total = len(df)
        for flag, sflag in zip(FLAGS, short_flags):
            n_true = int(df[flag].sum())
            rt = n_true / n_total * 100 if n_total > 0 else 0
            low = "LOW-N" if n_total < LOW_N else ""
            rows.append({"Flag": flag, "N_Users": n_total, "Count": n_true, "Rate_%": round(rt, 2), "Note": low})
        return pd.DataFrame(rows)
    else:
        rows = []
        for seg_val, grp in df.groupby(segment_col, observed=True):
            n = len(grp)
            row = {"Segment": str(seg_val)[:30] if pd.notna(seg_val) else "(null)", "N": n}
            for flag, sflag in zip(FLAGS, short_flags):
                row[sflag] = round(grp[flag].mean() * 100, 1)
            row["Note"] = "LOW-N" if n < LOW_N else ""
            rows.append(row)
        result_df = pd.DataFrame(rows)
        if show_aggregate and len(result_df) > 0:
            n_all = len(df)
            agg_row = {"Segment": "GRAND TOTAL", "N": n_all}
            for flag, sflag in zip(FLAGS, short_flags):
                agg_row[sflag] = round(df[flag].mean() * 100, 1)
            agg_row["Note"] = "← aggregate"
            result_df = pd.concat([result_df, pd.DataFrame([agg_row])], ignore_index=True)
        return result_df


def _build_time_table_df(df, segment_col=None):
    """Build a DataFrame of time-to percentiles."""
    short_tc = ["canvas", "build", "run", "ship"]

    def _row_data(grp, label):
        row = {"Segment": label, "N": len(grp)}
        low = "LOW-N" if len(grp) < LOW_N else ""
        for tc, st in zip(TIME_COLS, short_tc):
            vals = grp[tc].dropna()
            if len(vals) < 2:
                row[f"p25_{st}"] = None
                row[f"med_{st}"] = None
                row[f"p75_{st}"] = None
                row[f"p90_{st}"] = None
            else:
                row[f"p25_{st}"] = round(float(np.percentile(vals, 25)), 1)
                row[f"med_{st}"] = round(float(np.median(vals)), 1)
                row[f"p75_{st}"] = round(float(np.percentile(vals, 75)), 1)
                row[f"p90_{st}"] = round(float(np.percentile(vals, 90)), 1)
        row["Note"] = low
        return row

    rows = []
    if segment_col is None:
        rows.append(_row_data(df, "OVERALL"))
    else:
        for seg_val, grp in df.groupby(segment_col, observed=True):
            seg_str = str(seg_val)[:32] if pd.notna(seg_val) else "(null)"
            rows.append(_row_data(grp, seg_str))
    return pd.DataFrame(rows)


# ── Print helper: print title + formatted table (no tabulate dependency) ──────
def _print_table(title, df):
    SEP = "═" * 130
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: OVERALL FUNNEL
# ══════════════════════════════════════════════════════════════════════════════
_t1 = _build_rate_table_df(ladder)
_print_table("TABLE 1 — OVERALL ACTIVATION LADDER FUNNEL  (N = all 4,771 users)", _t1)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: SEGMENTED FLAG RATE TABLES
# ══════════════════════════════════════════════════════════════════════════════
segment_defs = [
    ("os",              "TABLE 2a — FLAG RATES BY OS",                                         False),
    ("browser",         "TABLE 2b — FLAG RATES BY BROWSER",                                    False),
    ("device_type",     "TABLE 2c — FLAG RATES BY DEVICE TYPE",                                True),
    ("country",         "TABLE 2d — FLAG RATES BY COUNTRY / GEO  [top 20 shown]",             False),
    ("referrer_top10",  "TABLE 2e — FLAG RATES BY ENTRY REFERRER DOMAIN (Top 10 + Other)",     False),
    ("product_surface", "TABLE 2f — FLAG RATES BY PRODUCT SURFACE / ENTRY PATH",               False),
]

for seg_col, title, show_agg in segment_defs:
    if seg_col == "country":
        top_segs = ladder[seg_col].value_counts().head(20).index
        _tbl = _build_rate_table_df(ladder[ladder[seg_col].isin(top_segs)], segment_col=seg_col, show_aggregate=show_agg)
    else:
        _tbl = _build_rate_table_df(ladder, segment_col=seg_col, show_aggregate=show_agg)
    _print_table(title, _tbl)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: TIME-TO PERCENTILES
# ══════════════════════════════════════════════════════════════════════════════
_t3a = _build_time_table_df(ladder)
_print_table("TABLE 3a — TIME-TO METRICS: OVERALL  [hours from t0, for users who reached each step]", _t3a)

for seg_col, title in [
    ("os",              "TABLE 3b — TIME-TO BY OS"),
    ("browser",         "TABLE 3c — TIME-TO BY BROWSER"),
    ("device_type",     "TABLE 3d — TIME-TO BY DEVICE TYPE"),
    ("country",         "TABLE 3e — TIME-TO BY COUNTRY / GEO  [top 20]"),
    ("referrer_top10",  "TABLE 3f — TIME-TO BY ENTRY REFERRER DOMAIN"),
    ("product_surface", "TABLE 3g — TIME-TO BY PRODUCT SURFACE"),
]:
    if seg_col == "country":
        top_segs = ladder[seg_col].value_counts().head(20).index
        _tbl = _build_time_table_df(ladder[ladder[seg_col].isin(top_segs)], segment_col=seg_col)
    else:
        _tbl = _build_time_table_df(ladder, segment_col=seg_col)
    _print_table(title, _tbl)

print("\n" + "═" * 80)
print("  ✅  ALL RATE TABLES COMPLETE")
print("  LOW-N flag = segments with fewer than 30 users")
print("═" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT TO CSV
# ══════════════════════════════════════════════════════════════════════════════
_csv_path = "rate_tables_full.csv"
ladder.to_csv(_csv_path, index=False)
print(f"✅  CSV exported → {_csv_path}  ({len(ladder):,} rows × {ladder.shape[1]} cols)")

# Helper functions kept as aliases for downstream compatibility
def print_rate_table(df, title, segment_col=None, show_aggregate=False):
    _tbl = _build_rate_table_df(df, segment_col, show_aggregate)
    _print_table(title, _tbl)

def print_time_table(df, title, segment_col=None):
    _tbl = _build_time_table_df(df, segment_col)
    _print_table(title, _tbl)
