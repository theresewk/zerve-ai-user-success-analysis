
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG             = "#1D1D20"
FG             = "#fbfbff"
FG2            = "#909094"
FRICTION_COLOR = "#D73027"
SUCCESS_COLOR  = "#1B9E77"
NEUTRAL_COLOR  = "#6B7280"
PERSONA_ORDER  = ["agent-led", "manual-led", "observer-only"]
LOW_N          = 30

def _base_layout(**kwargs):
    """Build a base Plotly layout dict, merge in chart-specific overrides."""
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG, family="Inter, Arial, sans-serif"),
        title_font=dict(color=FG, size=16),
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG, size=11)),
    )
    base.update(kwargs)
    return base

# ── Rebuild analysis df ────────────────────────────────────────────────────────
_USER        = "person_id"
_ladder_cols = [_USER, "run_within_7d", "runs_on_2plus_distinct_days_within_14d",
                "shipped", "run_within_24h"]
_ladder  = user_ladder_df[_ladder_cols].copy()
_persona = persona_df[[_USER, "persona"]].drop_duplicates(_USER)
_fa      = friction_features_df.drop(columns=["persona"], errors="ignore")
viz_analysis_df = _fa.merge(_ladder, on=_USER, how="left").merge(_persona, on=_USER, how="left")

# Gating group
_has_any_credits = (viz_analysis_df["credits_warn_count_7d"] > 0) | \
                   (viz_analysis_df["credits_exceeded_count_7d"] > 0)
_before = viz_analysis_df["credits_friction_before_run_7d"] == True
_after  = (~_before) & _has_any_credits
viz_analysis_df["_gating_group"] = "no_friction"
viz_analysis_df.loc[_after,  "_gating_group"] = "friction_after_first_run"
viz_analysis_df.loc[_before, "_gating_group"] = "friction_before_first_run"

print(f"viz_analysis_df: {viz_analysis_df.shape[0]:,} rows × {viz_analysis_df.shape[1]} cols")
print(f"credits_cohort_7d distribution:")
_coh_dist = viz_analysis_df['credits_cohort_7d'].value_counts().reset_index()
_coh_dist.columns = ["cohort", "count"]
print(_coh_dist.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Grouped bar: success rates by credits cohort
# ══════════════════════════════════════════════════════════════════════════════
_cohort_order = ["no_friction", "warn_only", "exceeded"]
_metrics_cfg  = [
    ("run_within_7d",                           "Activate (Run 7d)", SUCCESS_COLOR),
    ("runs_on_2plus_distinct_days_within_14d",  "Habit (2+ days)",   "#8DE5A1"),
    ("shipped",                                 "Ship",              NEUTRAL_COLOR),
]

_cohort_rows = []
for _coh in _cohort_order:
    _sub = viz_analysis_df[viz_analysis_df["credits_cohort_7d"] == _coh]
    _n   = len(_sub)
    _row = {"cohort": _coh, "n": _n}
    for _col, _lbl, _ in _metrics_cfg:
        _row[_col] = _sub[_col].mean() * 100 if _n > 0 else np.nan
    _cohort_rows.append(_row)
_cohort_df = pd.DataFrame(_cohort_rows)

print(f"\nCohort success rates:")
print(_cohort_df.to_string(index=False))

_x_labels = [c.replace("_", " ").title() for c in _cohort_order]
fig_credits_bar = go.Figure()

for _col, _lbl, _clr in _metrics_cfg:
    _vals = [_cohort_df.loc[_cohort_df["cohort"] == c, _col].values[0] for c in _cohort_order]
    _text = [f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in _vals]
    fig_credits_bar.add_trace(go.Bar(
        name=_lbl, x=_x_labels, y=_vals,
        text=_text, textposition="outside",
        textfont=dict(color=FG, size=11),
        marker_color=_clr,
        marker_line=dict(color=BG, width=1.5),
        opacity=0.90,
    ))

for _idx, _row in enumerate(_cohort_rows):
    _lw = " ⚠" if _row["n"] < LOW_N else ""
    fig_credits_bar.add_annotation(
        x=_x_labels[_idx], y=-9,
        text=f"N={_row['n']:,}{_lw}",
        showarrow=False,
        font=dict(color=FG2, size=10),
        xref="x", yref="y",
    )

fig_credits_bar.update_layout(
    _base_layout(
        title="Success Rates by Credits Cohort (7d window)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=FG), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Success Rate (%)", range=[0, 120],
                   gridcolor="#333337", zerolinecolor="#333337"),
        xaxis=dict(title="Credits Cohort"),
    )
)
fig_credits_bar.show()
print("✓ Chart 1: Success rates by credits cohort")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Pre-run gating impact
# ══════════════════════════════════════════════════════════════════════════════
_gating_order  = ["no_friction", "friction_before_first_run", "friction_after_first_run"]
_gating_labels = ["No Friction", "Friction Before Run", "Friction After Run"]
_gating_metrics = [
    ("run_within_24h",                         "Run in 24h (%)"),
    ("run_within_7d",                          "Activate – Run 7d (%)"),
    ("runs_on_2plus_distinct_days_within_14d", "Habit (%)"),
    ("shipped",                                "Ship (%)"),
]

_gate_rows = []
for _gcat, _glbl in zip(_gating_order, _gating_labels):
    _sub = viz_analysis_df[viz_analysis_df["_gating_group"] == _gcat]
    _n   = len(_sub)
    _lw  = " ⚠" if _n < LOW_N else ""
    _row = {"Group": _glbl, "N": f"{_n:,}{_lw}"}
    for _mc, _mv in _gating_metrics:
        _v = _sub[_mc].mean() * 100 if _n > 0 else np.nan
        _row[_mv] = f"{_v:.1f}%" if not np.isnan(_v) else "—"
    _gate_rows.append(_row)

_gate_df = pd.DataFrame(_gate_rows)
print(f"\nPre-run gating impact:")
print(_gate_df.to_string(index=False))

_gh  = list(_gate_df.columns)
_gv  = [_gate_df[c].tolist() for c in _gh]
_row_fills = ["rgba(107,114,128,0.2)", "rgba(215,48,39,0.2)", "rgba(255,180,130,0.2)"]

fig_gating_table = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in _gh],
        fill_color="#2A2A2E", align="center",
        font=dict(color=FG, size=12), line_color="#444448", height=36,
    ),
    cells=dict(
        values=_gv,
        fill_color=[
            _row_fills,
            ["#2A2A2E"] * 3,
            *[["#1D1D20"] * 3 for _ in _gating_metrics],
        ],
        align=["left", "right"] + ["center"] * len(_gating_metrics),
        font=dict(color=FG, size=11),
        line_color="#333337", height=32,
    ),
)])
fig_gating_table.update_layout(
    _base_layout(
        title="Pre-Run Gating Impact on Success Metrics",
        margin=dict(l=20, r=20, t=60, b=20),
        height=210,
    )
)
fig_gating_table.show()
print("✓ Table 2: Pre-run gating impact table")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Friction prevalence heatmap by persona
# ══════════════════════════════════════════════════════════════════════════════
_heatmap_features = [
    ("credits_warn_count_7d",          "Credits Warn"),
    ("credits_exceeded_count_7d",      "Credits Exceeded"),
    ("credits_friction_before_run_7d", "Before-Run Gate"),
    ("stop_count_7d",                  "Execution Stop"),
    ("any_seats_friction_7d",          "Seats Friction"),
    ("agent_error_assist_count_7d",    "Agent Errors"),
]
_feat_labels = [lbl for _, lbl in _heatmap_features]
_hmap_data   = []
_persona_ns  = []

for _p in PERSONA_ORDER:
    _sub = viz_analysis_df[viz_analysis_df["persona"] == _p]
    _persona_ns.append(len(_sub))
    _row_rates = []
    for _col, _ in _heatmap_features:
        _s = _sub[_col]
        _rate = _s.mean() * 100 if _s.dtype == bool else (_s > 0).mean() * 100
        _row_rates.append(round(float(_rate), 1))
    _hmap_data.append(_row_rates)

_hmap_arr = np.array(_hmap_data)
_y_labels = [f"{p}<br><sub>N={n:,}</sub>" for p, n in zip(PERSONA_ORDER, _persona_ns)]

print(f"\nFriction prevalence heatmap data:")
_hmap_rows = [
    {"persona": _p, "N": _n, **{lbl: _r for lbl, _r in zip(_feat_labels, _rd)}}
    for _p, _n, _rd in zip(PERSONA_ORDER, _persona_ns, _hmap_data)
]
print(pd.DataFrame(_hmap_rows).to_string(index=False))

fig_heatmap = go.Figure(data=go.Heatmap(
    z=_hmap_arr,
    x=_feat_labels,
    y=_y_labels,
    colorscale=[
        [0.0,  "#1D1D20"],
        [0.25, "#6B7280"],
        [0.55, "#FFB482"],
        [1.0,  "#D73027"],
    ],
    zmin=0,
    zmax=max(float(_hmap_arr.max()), 1.0),
    colorbar=dict(
        title=dict(text="Prevalence (%)", font=dict(color=FG2, size=11)),
        tickfont=dict(color=FG2),
        outlinecolor="#333337", outlinewidth=1,
    ),
    text=[[f"{v:.1f}%" for v in _rd] for _rd in _hmap_data],
    texttemplate="%{text}",
    textfont=dict(size=12, color=FG),
    hoverongaps=False,
))
fig_heatmap.update_layout(
    _base_layout(
        title="Friction Prevalence by Persona (% of users, 7d window)",
        xaxis=dict(title="", tickfont=dict(color=FG, size=11), tickangle=-20),
        yaxis=dict(title="", tickfont=dict(color=FG, size=11)),
        height=340,
        margin=dict(l=130, r=80, t=70, b=60),
    )
)
fig_heatmap.show()
print("✓ Chart 3: Friction prevalence heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Warn-to-add-credits timing distribution
# ══════════════════════════════════════════════════════════════════════════════
_delta_all = viz_analysis_df["credits_warn_to_intent_hours_7d"].dropna()
_delta_all = _delta_all[_delta_all >= 0]
print(f"\nWarn-to-intent users (7d, >=0h): {len(_delta_all):,}")

def _timing_row(series, label):
    _s  = series.dropna()
    _s  = _s[_s >= 0]
    _n  = len(_s)
    _lw = " ⚠" if _n < LOW_N else ""
    if _n >= 2:
        return {
            "Persona / Group":  label,
            "N (w/ timing)":    f"{_n:,}{_lw}",
            "P25 (h)":          f"{np.percentile(_s, 25):.2f}",
            "Median P50 (h)":   f"{np.percentile(_s, 50):.2f}",
            "P75 (h)":          f"{np.percentile(_s, 75):.2f}",
            "P90 (h)":          f"{np.percentile(_s, 90):.2f}",
        }
    return {
        "Persona / Group": label,
        "N (w/ timing)":   f"{_n:,}{_lw}",
        "P25 (h)": "—", "Median P50 (h)": "—", "P75 (h)": "—", "P90 (h)": "—",
    }

_timing_rows = [_timing_row(_delta_all, "OVERALL")]
for _p in PERSONA_ORDER:
    _ps = viz_analysis_df[viz_analysis_df["persona"] == _p]["credits_warn_to_intent_hours_7d"]
    _timing_rows.append(_timing_row(_ps, _p))

_timing_df = pd.DataFrame(_timing_rows)
print(f"\nWarn-to-add-credits timing distribution:")
print(_timing_df.to_string(index=False))

_th      = list(_timing_df.columns)
_tv      = [_timing_df[c].tolist() for c in _th]
_n_rows  = len(_timing_rows)
_row_fill = [
    ["rgba(215,48,39,0.15)" if _k == 0 else "#1D1D20" for _k in range(_n_rows)]
    for _ in _th
]

fig_timing_table = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in _th],
        fill_color="#2A2A2E", align="center",
        font=dict(color=FG, size=12), line_color="#444448", height=36,
    ),
    cells=dict(
        values=_tv,
        fill_color=_row_fill,
        align=["left", "right", "center", "center", "center", "center"],
        font=dict(
            color=[
                [FRICTION_COLOR if _k == 0 else FG for _k in range(_n_rows)],
                *[[FG] * _n_rows for _ in range(len(_th) - 1)],
            ],
            size=11,
        ),
        line_color="#333337", height=32,
    ),
)])
fig_timing_table.update_layout(
    _base_layout(
        title="Warn-to-Add-Credits Timing Distribution (7d window)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=240,
    )
)
fig_timing_table.show()
print("✓ Table 4: Timing distribution")

print("\n✅  All 4 credits friction visualizations complete")
