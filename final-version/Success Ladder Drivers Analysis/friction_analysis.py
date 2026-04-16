
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Design System ──────────────────────────────────────────────────────────────
BG       = "#1D1D20"
FG       = "#fbfbff"
FG2      = "#909094"
BLUE     = "#A1C9F4"
ORANGE   = "#FFB482"
GREEN    = "#8DE5A1"
CORAL    = "#FF9F9B"
LAVENDER = "#D0BBFF"
YELLOW   = "#ffd400"
ACCENT   = "#17b26a"
LOW_N    = 30

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG2, "ytick.color": FG2,
    "axes.edgecolor": FG2, "grid.color": "#333337",
    "font.family": "sans-serif", "font.size": 11,
})

_USER = "person_id"

# ── Join tables ────────────────────────────────────────────────────────────────
_ladder_cols = [
    _USER,
    "run_within_7d",
    "runs_on_2plus_distinct_days_within_14d",
    "shipped",
    "run_within_24h",
]
_ladder = user_ladder_df[_ladder_cols].copy()
_persona = persona_df[[_USER, "persona"]].drop_duplicates(_USER)

_fa = friction_features_df.drop(columns=["persona"], errors="ignore")
analysis_df = _fa.merge(_ladder, on=_USER, how="left").merge(_persona, on=_USER, how="left")

N_TOTAL = len(analysis_df)
PERSONA_ORDER = ["agent-led", "manual-led", "observer-only"]

def _flag_low(n, threshold=LOW_N):
    return "⚠" if n < threshold else ""

SEP = "═" * 80

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FRICTION PREVALENCE BY TIER
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SECTION 1 — FRICTION PREVALENCE BY SUCCESS TIER (7d window)")
print(SEP)

TIERS = {
    "Activate (run_within_7d)":                   "run_within_7d",
    "Habit (runs_2plus_distinct_days_14d)":        "runs_on_2plus_distinct_days_within_14d",
    "Ship (shipped)":                              "shipped",
}

FRICTION_FEATURES_7D = [
    ("credits_warn_count_7d",          "Credits Warning"),
    ("credits_exceeded_count_7d",      "Credits Exceeded"),
    ("credits_add_intent_count_7d",    "Add-Credits Intent"),
    ("credits_friction_before_run_7d", "Credits Before First Run"),
    ("stop_count_7d",                  "Execution Stop"),
    ("any_seats_friction_7d",          "Seats Friction"),
    ("agent_error_assist_count_7d",    "Agent Error Assist"),
    ("any_agent_error_friction_7d",    "Any Agent Error"),
]

# Build binary friction flags
for col, _ in FRICTION_FEATURES_7D:
    s = analysis_df[col]
    if s.dtype == bool:
        analysis_df[f"_ff_{col}"] = s.astype(int)
    else:
        analysis_df[f"_ff_{col}"] = (s > 0).astype(int)

section1_rows = []
for col, label in FRICTION_FEATURES_7D:
    ff = f"_ff_{col}"
    row_data = {"Feature": label}
    for tier_label, tier_col in TIERS.items():
        tier_yes = analysis_df[analysis_df[tier_col] == True]
        tier_no  = analysis_df[analysis_df[tier_col] == False]
        n_yes = len(tier_yes)
        n_no  = len(tier_no)
        fric_yes = tier_yes[ff].sum()
        fric_no  = tier_no[ff].sum()
        rate_yes = fric_yes / n_yes * 100 if n_yes > 0 else 0
        short = tier_label.split("(")[0].strip()
        row_data[f"{short}_yes_n"]   = int(fric_yes)
        row_data[f"{short}_no_n"]    = int(fric_no)
        row_data[f"{short}_yes_rate%"] = round(rate_yes, 1)
    section1_rows.append(row_data)

section1_df = pd.DataFrame(section1_rows)
print(f"\n  Tier sizes: " + "  |  ".join(
    f"{t.split('(')[0].strip()}: {analysis_df[c].sum():,}" for t, c in TIERS.items()
))
print(f"  Total users: {N_TOTAL:,}")
print()
print(section1_df.to_string(index=False))
print(f"\n  ⚠ = fewer than {LOW_N} users in segment")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SUCCESS RATES BY FRICTION COHORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SECTION 2 — SUCCESS RATES BY FRICTION COHORT (7d window)")
print(SEP)

COHORT_CONFIGS = [
    {
        "name": "Credits Cohort (credits_cohort_7d)",
        "col":  "credits_cohort_7d",
        "cats": ["no_friction", "warn_only", "exceeded"],
    },
    {
        "name": "Stop-Disruption Cohort (stop_count_7d)",
        "col":  "_derived_stop_cohort",
        "cats": ["no_stops", "has_stops"],
    },
    {
        "name": "Seats Cohort (any_seats_friction_7d)",
        "col":  "_derived_seats_cohort",
        "cats": ["no_seats_friction", "has_seats_friction"],
    },
    {
        "name": "Agent-Error Cohort (any_agent_error_friction_7d)",
        "col":  "_derived_agent_cohort",
        "cats": ["no_agent_errors", "has_agent_errors"],
    },
]

# Derive binary cohort columns
analysis_df["_derived_stop_cohort"]  = np.where(analysis_df["stop_count_7d"] > 0, "has_stops", "no_stops")
analysis_df["_derived_seats_cohort"] = np.where(analysis_df["any_seats_friction_7d"], "has_seats_friction", "no_seats_friction")
analysis_df["_derived_agent_cohort"] = np.where(analysis_df["any_agent_error_friction_7d"], "has_agent_errors", "no_agent_errors")

section2_tables = {}
for cfg in COHORT_CONFIGS:
    print(f"\n  ── {cfg['name']} ──")
    rows = []
    for cat in cfg["cats"]:
        sub = analysis_df[analysis_df[cfg["col"]] == cat]
        n   = len(sub)
        act = sub["run_within_7d"].mean() * 100 if n > 0 else np.nan
        hab = sub["runs_on_2plus_distinct_days_within_14d"].mean() * 100 if n > 0 else np.nan
        shp = sub["shipped"].mean() * 100 if n > 0 else np.nan
        lw  = _flag_low(n)
        rows.append({"cohort": cat, "n": n, "activate_%": round(act, 1), "habit_%": round(hab, 1), "ship_%": round(shp, 1), "note": lw})
    _s2_df = pd.DataFrame(rows)
    section2_tables[cfg["name"]] = _s2_df
    print(_s2_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRE-RUN FRICTION GATING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SECTION 3 — PRE-RUN FRICTION GATING")
print(SEP)

_has_any_credits = (analysis_df["credits_warn_count_7d"] > 0) | (analysis_df["credits_exceeded_count_7d"] > 0)
_before = analysis_df["credits_friction_before_run_7d"] == True
_after  = (~_before) & _has_any_credits

analysis_df["_gating_group"] = "no_friction"
analysis_df.loc[_after,  "_gating_group"] = "friction_after_first_run"
analysis_df.loc[_before, "_gating_group"] = "friction_before_first_run"

GATING_CATS = ["no_friction", "friction_before_first_run", "friction_after_first_run"]
GATING_METRICS = {
    "run_within_24h": "Run in 24h",
    "run_within_7d":  "Run in 7d (Activate)",
    "runs_on_2plus_distinct_days_within_14d": "Habit",
    "shipped": "Ship",
}

section3_rows = []
for cat in GATING_CATS:
    sub = analysis_df[analysis_df["_gating_group"] == cat]
    n   = len(sub)
    lw  = _flag_low(n)
    row = {"Group": cat, "N": n, "Note": lw}
    for mc, mv in GATING_METRICS.items():
        r = sub[mc].mean() * 100 if n > 0 else np.nan
        row[mv] = round(r, 1)
    section3_rows.append(row)

section3_df = pd.DataFrame(section3_rows)
print()
print(section3_df.to_string(index=False))
print(f"\n  ⚠ = fewer than {LOW_N} users in segment")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TIME-TO-ADD-CREDITS DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SECTION 4 — TIME-TO-ADD-CREDITS DISTRIBUTION (warn → intent, 7d window)")
print(SEP)

_delta = analysis_df["credits_warn_to_intent_hours_7d"].dropna()
_delta = _delta[_delta >= 0]

print(f"\n  Users with warn→intent delta: {len(_delta):,}")

if len(_delta) >= 5:
    p25 = float(np.percentile(_delta, 25))
    p50 = float(np.percentile(_delta, 50))
    p75 = float(np.percentile(_delta, 75))
    p90 = float(np.percentile(_delta, 90))

    section4_rows = [{"Group": f"OVERALL (N={len(_delta):,})", "P25_h": round(p25, 4),
                      "Median_h": round(p50, 4), "P75_h": round(p75, 4), "P90_h": round(p90, 4)}]
    for p in PERSONA_ORDER:
        sub_p = analysis_df[analysis_df["persona"] == p]["credits_warn_to_intent_hours_7d"].dropna()
        sub_p = sub_p[sub_p >= 0]
        n_p   = len(sub_p)
        lw    = _flag_low(n_p)
        if n_p >= 2:
            section4_rows.append({
                "Group": f"{p} (N={n_p:,}){lw}",
                "P25_h": round(float(np.percentile(sub_p, 25)), 4),
                "Median_h": round(float(np.percentile(sub_p, 50)), 4),
                "P75_h": round(float(np.percentile(sub_p, 75)), 4),
                "P90_h": round(float(np.percentile(sub_p, 90)), 4),
            })
        else:
            section4_rows.append({"Group": f"{p} (N={n_p:,}){lw}", "P25_h": None, "Median_h": None, "P75_h": None, "P90_h": None})

    section4_df = pd.DataFrame(section4_rows)
    print()
    print(section4_df.to_string(index=False))
else:
    print(f"  ⚠ Insufficient data for distribution (N={len(_delta)})")
    _avail = analysis_df[analysis_df["credits_warn_to_intent_hours_7d"].notna()]
    if len(_avail) > 0:
        _avail_rows = []
        for _, r in _avail.iterrows():
            h = r["credits_warn_to_intent_hours_7d"]
            if h >= 0:
                _avail_rows.append({"persona": r["persona"], "hours": round(h, 4)})
        if _avail_rows:
            print(pd.DataFrame(_avail_rows).to_string(index=False))
    section4_df = pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SECTION 5 — VISUALIZATIONS")
print(SEP)

# ── VIZ 1: Success Rates by Credits Cohort ──
credits_data = section2_tables["Credits Cohort (credits_cohort_7d)"]
cohorts = credits_data["cohort"].tolist()
metrics = ["activate_%", "habit_%", "ship_%"]
metric_labels = ["Activate (Run 7d)", "Habit (2+ days)", "Ship"]
colors = [BLUE, GREEN, CORAL]

x = np.arange(len(cohorts))
width = 0.25

success_rates_by_credits_cohort = plt.figure(figsize=(11, 6))
success_rates_by_credits_cohort.patch.set_facecolor(BG)
ax1 = success_rates_by_credits_cohort.add_subplot(111)
ax1.set_facecolor(BG)

for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    vals = credits_data[metric].tolist()
    bars = ax1.bar(x + i * width, vals, width, label=label, color=color, alpha=0.88, zorder=3)
    for bar, val in zip(bars, vals):
        if not (val is None or (isinstance(val, float) and np.isnan(val))) and val > 0.5:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=FG)

ax1.set_xticks(x + width)
ax1.set_xticklabels([c.replace("_", " ").title() for c in cohorts], fontsize=11, color=FG)
ax1.set_ylabel("Success Rate (%)", color=FG, fontsize=12)
ax1.set_title("Success Rates by Credits Cohort", color=FG, fontsize=14, fontweight="bold", pad=14)
ax1.legend(framealpha=0.15, facecolor=BG, labelcolor=FG, fontsize=10)
ax1.yaxis.grid(True, alpha=0.2, zorder=0)
_max_val = max((v for v in credits_data[metrics].values.flatten() if v is not None and not np.isnan(v)), default=5)
ax1.set_ylim(0, max(_max_val * 1.22, 5))

for i, (_, row) in enumerate(credits_data.iterrows()):
    lw = " ⚠" if row["n"] < LOW_N else ""
    ax1.text(i + width, -4, f"N={row['n']:,}{lw}", ha="center", va="top", fontsize=8.5, color=FG2,
             transform=ax1.transData)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
print("  ✓ Chart 1: Success rates by credits cohort")

# ── VIZ 2: Stop-After-Run Rate by Persona ──
stop_persona_rows = []
for p in PERSONA_ORDER:
    sub_p = analysis_df[analysis_df["persona"] == p]
    n_p   = len(sub_p)
    sar   = sub_p["stop_after_run_rate_7d"].dropna()
    rate  = sar.mean() * 100 if len(sar) > 0 else 0.0
    stop_persona_rows.append({"persona": p, "n": n_p, "n_with_rate": len(sar), "stop_after_run_pct": rate})

stop_by_persona_df = pd.DataFrame(stop_persona_rows)

stop_after_run_by_persona = plt.figure(figsize=(9, 5))
stop_after_run_by_persona.patch.set_facecolor(BG)
ax2 = stop_after_run_by_persona.add_subplot(111)
ax2.set_facecolor(BG)

_colors_bar = [BLUE, ORANGE, LAVENDER]
_bars = ax2.bar(stop_by_persona_df["persona"],
                stop_by_persona_df["stop_after_run_pct"],
                color=_colors_bar, alpha=0.88, zorder=3, width=0.55)

for bar, row in zip(_bars, stop_by_persona_df.itertuples()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f"{row.stop_after_run_pct:.1f}%\nN={row.n_with_rate:,}",
             ha="center", va="bottom", fontsize=10, color=FG)

ax2.set_ylabel("Avg Stop-After-Run Rate (%)", color=FG, fontsize=12)
ax2.set_title("Stop-After-Run Rate by Persona", color=FG, fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("Persona", color=FG, fontsize=11)
ax2.yaxis.grid(True, alpha=0.2, zorder=0)
ax2.set_ylim(0, max(stop_by_persona_df["stop_after_run_pct"].max() * 1.35, 1))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
print("  ✓ Chart 2: Stop-after-run rate by persona")

# ── VIZ 3: Friction Feature Prevalence Heatmap by Persona ──
HEATMAP_FEATURES = [
    ("credits_warn_count_7d",          "Credits Warning"),
    ("credits_exceeded_count_7d",      "Credits Exceeded"),
    ("credits_friction_before_run_7d", "Credits Before Run"),
    ("stop_count_7d",                  "Execution Stop"),
    ("any_seats_friction_7d",          "Seats Friction"),
    ("any_agent_error_friction_7d",    "Agent Error"),
]

heatmap_data = {}
heatmap_ns   = {}
for p in PERSONA_ORDER:
    sub_p = analysis_df[analysis_df["persona"] == p]
    heatmap_ns[p] = len(sub_p)
    col_rates = []
    for col, _ in HEATMAP_FEATURES:
        s = sub_p[col]
        if s.dtype == bool:
            rate_pct = s.mean() * 100
        else:
            rate_pct = (s > 0).mean() * 100
        col_rates.append(round(rate_pct, 1))
    heatmap_data[p] = col_rates

hmap_array = np.array([heatmap_data[p] for p in PERSONA_ORDER])  # (3, 6)
feat_labels = [lbl for _, lbl in HEATMAP_FEATURES]

friction_prevalence_heatmap = plt.figure(figsize=(11, 5))
friction_prevalence_heatmap.patch.set_facecolor(BG)
ax3 = friction_prevalence_heatmap.add_subplot(111)
ax3.set_facecolor(BG)

im = ax3.imshow(hmap_array, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(hmap_array.max(), 1))

ax3.set_xticks(range(len(feat_labels)))
ax3.set_xticklabels(feat_labels, rotation=28, ha="right", fontsize=10, color=FG)
ax3.set_yticks(range(len(PERSONA_ORDER)))
ax3.set_yticklabels(
    [f"{p}\n(N={heatmap_ns[p]:,})" for p in PERSONA_ORDER],
    fontsize=11, color=FG
)

for i in range(len(PERSONA_ORDER)):
    for j in range(len(feat_labels)):
        val = hmap_array[i, j]
        text_color = "#1D1D20" if val > hmap_array.max() * 0.55 else FG
        ax3.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=10.5,
                 fontweight="bold", color=text_color)

cbar = friction_prevalence_heatmap.colorbar(im, ax=ax3, fraction=0.03, pad=0.03)
cbar.set_label("Prevalence (%)", color=FG, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=FG2)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG2)

ax3.set_title("Friction Feature Prevalence by Persona", color=FG, fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
print("  ✓ Chart 3: Friction feature prevalence heatmap by persona")

print(f"\n{SEP}")
print("  ✅  ALL 5 SECTIONS COMPLETE")
print(f"  Total users analysed: {N_TOTAL:,}")
print(f"  Low-N threshold: {LOW_N} users")
print(SEP)
