
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Design system ──────────────────────────────────────────────────────────────
BG        = "#1D1D20"
FG        = "#fbfbff"
FG2       = "#909094"
SUCCESS   = "#1B9E77"
NEUTRAL   = "#6B7280"
BLUE      = "#A1C9F4"
ORANGE    = "#FFB482"
GREEN     = "#8DE5A1"
CORAL     = "#FF9F9B"
LAVENDER  = "#D0BBFF"
YELLOW    = "#ffd400"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FG, family="Inter, system-ui, sans-serif"),
    margin=dict(l=60, r=40, t=70, b=60),
)

# ── Constants ──────────────────────────────────────────────────────────────────
_FLAGS = [
    "run_within_24h",
    "run_within_7d",
    "run_within_first_session",
    "runs_on_2plus_distinct_days_within_14d",
    "runs_in_3plus_distinct_sessions_within_14d",
    "returned_after_day7",
    "shipped",
    "ship_then_use",
]

_FLAG_LABELS = [
    "Run within 24 h",
    "Run within 7 d",
    "Run in first session",
    "Runs on 2+ days (14 d)",
    "Runs in 3+ sessions (14 d)",
    "Returned after day 7",
    "Shipped",
    "Ship → Use",
]

_TIME_COLS = [
    "time_to_first_canvas",
    "time_to_first_build",
    "time_to_first_run",
    "time_to_first_ship",
]
_TIME_LABELS = ["Canvas", "Build", "Run", "Ship"]

_SEG_COLS = [
    ("os",              "OS"),
    ("browser",         "Browser"),
    ("device_type",     "Device Type"),
    ("country",         "Country"),
    ("referrer_top10",  "Referrer"),
    ("product_surface", "Surface"),
]

LOW_N = 30

_n_total = len(ladder)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Horizontal funnel bar chart
# ══════════════════════════════════════════════════════════════════════════════
_flag_rates = [ladder[f].mean() * 100 for f in _FLAGS]

_order = np.argsort(_flag_rates)[::-1]
_sorted_labels = [_FLAG_LABELS[i] for i in _order]
_sorted_rates  = [_flag_rates[i] for i in _order]
_sorted_counts = [int(ladder[_FLAGS[i]].sum()) for i in _order]

_bar_colors = [
    SUCCESS if r >= 5 else NEUTRAL
    for r in _sorted_rates
]

funnel_chart = go.Figure()
funnel_chart.add_trace(go.Bar(
    y=_sorted_labels,
    x=_sorted_rates,
    orientation="h",
    marker_color=_bar_colors,
    text=[f"{r:.1f}%  ({c:,})" for r, c in zip(_sorted_rates, _sorted_counts)],
    textposition="outside",
    textfont=dict(color=FG, size=12),
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>Rate: %{x:.2f}%<extra></extra>",
))

funnel_chart.update_layout(
    **_PLOTLY_LAYOUT,
    title=dict(
        text=f"Activation Success Ladder  — % of all {_n_total:,} users",
        font=dict(size=18, color=FG),
        x=0.0,
    ),
    xaxis=dict(
        title="% of Total Users",
        ticksuffix="%",
        gridcolor="#2e2e33",
        color=FG2,
        range=[0, max(_sorted_rates) * 1.35],
    ),
    yaxis=dict(
        color=FG2,
        tickfont=dict(size=13),
    ),
    height=480,
    showlegend=False,
)

print("✅ Chart 1: Funnel bar chart rendered")
funnel_chart.show()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Time-to-milestone
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 80)
print("  TIME-TO-MILESTONE  (hours from t0 — positive values only)")
print("═" * 80)

_time_rows = []
for _tc, _tl in zip(_TIME_COLS, _TIME_LABELS):
    _vals = ladder[_tc].dropna()
    _vals = _vals[_vals > 0]
    if len(_vals) >= 2:
        _p25 = float(np.percentile(_vals, 25))
        _p50 = float(np.percentile(_vals, 50))
        _p75 = float(np.percentile(_vals, 75))
        _p90 = float(np.percentile(_vals, 90))
    else:
        _p25 = _p50 = _p75 = _p90 = float("nan")
    _time_rows.append({
        "Milestone": _tl,
        "N (reached)": len(_vals),
        "P25 (h)": round(_p25, 2),
        "Median (h)": round(_p50, 2),
        "P75 (h)": round(_p75, 2),
        "P90 (h)": round(_p90, 2),
    })

time_milestone_df = pd.DataFrame(_time_rows)
print()
print(time_milestone_df.to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Segment breakdown charts + tables
# ══════════════════════════════════════════════════════════════════════════════

def _make_segment_chart(df, seg_col, seg_title):
    """Build a grouped bar chart: run_within_7d & shipped rates by segment."""
    if seg_col == "country":
        _top = df[seg_col].value_counts().head(15).index
        df = df[df[seg_col].isin(_top)]

    _grp = (
        df.groupby(seg_col)[_FLAGS]
        .agg(["mean", "count"])
        .reset_index()
    )
    _grp.columns = [
        f"{a}_{b}" if b else a
        for a, b in _grp.columns
    ]
    _n_col = f"{_FLAGS[0]}_count"
    _grp = _grp[_grp[_n_col] >= LOW_N].copy()
    _grp = _grp.sort_values(f"run_within_7d_mean", ascending=False)

    if len(_grp) == 0:
        return None

    _segs   = _grp[seg_col].astype(str).tolist()
    _rate7d  = (_grp["run_within_7d_mean"] * 100).round(1).tolist()
    _shipped = (_grp["shipped_mean"] * 100).round(1).tolist()
    _ns      = _grp[_n_col].astype(int).tolist()

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name="Run within 7d",
        x=_segs,
        y=_rate7d,
        marker_color=SUCCESS,
        text=[f"{v:.1f}%" for v in _rate7d],
        textposition="outside",
        textfont=dict(color=FG, size=11),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Run 7d: %{y:.1f}%<extra></extra>",
    ))
    _fig.add_trace(go.Bar(
        name="Shipped",
        x=_segs,
        y=_shipped,
        marker_color=NEUTRAL,
        text=[f"{v:.1f}%" for v in _shipped],
        textposition="outside",
        textfont=dict(color=FG, size=11),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Shipped: %{y:.1f}%<extra></extra>",
    ))

    _max_y = max(max(_rate7d, default=0), max(_shipped, default=0))
    _fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(
            text=f"Success Rates by {seg_title}",
            font=dict(size=16, color=FG),
            x=0.0,
        ),
        barmode="group",
        xaxis=dict(
            color=FG2,
            tickangle=-30 if len(_segs) > 6 else 0,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="% of Users",
            ticksuffix="%",
            gridcolor="#2e2e33",
            color=FG2,
            range=[0, min(_max_y * 1.4, 100)],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=FG),
        ),
        height=420,
    )
    return _fig

# Segment breakdown tables + charts
print("═" * 80)
print("  SEGMENT BREAKDOWN — Success rates by dimension")
print("═" * 80)

_seg_chart_list = []
for _sc, _st in _SEG_COLS:
    print(f"\n── {_st} ──")
    if _sc == "country":
        _top20 = ladder[_sc].value_counts().head(20).index
        _sub_seg = ladder[ladder[_sc].isin(_top20)]
    else:
        _sub_seg = ladder

    _tbl_rows = []
    for _sv, _g in _sub_seg.groupby(_sc, observed=True):
        if len(_g) < LOW_N:
            continue
        _row_d = {"Segment": str(_sv)[:30], "N": len(_g)}
        for _f, _fl in zip(_FLAGS, _FLAG_LABELS):
            _row_d[_fl] = round(_g[_f].mean() * 100, 1)
        _tbl_rows.append(_row_d)

    if _tbl_rows:
        _seg_tbl = pd.DataFrame(_tbl_rows).sort_values("N", ascending=False)
        print(_seg_tbl.to_string(index=False))
    else:
        print("  (no segments with N >= 30)")

    _chart = _make_segment_chart(ladder.copy(), _sc, _st)
    if _chart is not None:
        _seg_chart_list.append((_st, _chart))

# Render all segment charts
for _st, _chart in _seg_chart_list:
    print(f"\n✅ Segment chart: {_st}")
    _chart.show()

print("\n" + "═" * 80)
print("  ✅  viz_success_ladder: all charts and tables complete")
print("═" * 80)
