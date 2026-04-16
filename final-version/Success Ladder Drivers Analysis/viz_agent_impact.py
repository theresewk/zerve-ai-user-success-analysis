
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = "#1D1D20"
FG      = "#fbfbff"
FG2     = "#909094"
COL_AGENT   = "#6A3D9A"   # purple  — agent-led
COL_MANUAL  = "#1F78B4"   # blue    — manual-led
COL_OBS     = "#6B7280"   # grey    — observer-only
PERSONA_COLORS = {
    "agent-led":    COL_AGENT,
    "manual-led":   COL_MANUAL,
    "observer-only": COL_OBS,
}
PERSONA_ORDER = ["agent-led", "manual-led", "observer-only"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=FG, family="Inter, Arial, sans-serif", size=13),
        title=dict(font=dict(color=FG, size=16, family="Inter, Arial, sans-serif")),
        xaxis=dict(gridcolor="#2e2e34", zerolinecolor="#2e2e34", tickfont=dict(color=FG2)),
        yaxis=dict(gridcolor="#2e2e34", zerolinecolor="#2e2e34", tickfont=dict(color=FG2)),
        legend=dict(bgcolor=BG, font=dict(color=FG), borderwidth=0),
        margin=dict(l=60, r=40, t=80, b=60),
    )
)
pio.templates["zerve"] = PLOTLY_TEMPLATE
pio.templates.default = "zerve"

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — PERSONA DISTRIBUTION DONUT
# ══════════════════════════════════════════════════════════════════════════════
_total_users = len(persona_df)
_pcounts = persona_df["persona"].value_counts()
_labels  = [p for p in PERSONA_ORDER if p in _pcounts.index]
_values  = [_pcounts[p] for p in _labels]
_colors  = [PERSONA_COLORS[p] for p in _labels]
_pcts    = [v / _total_users * 100 for v in _values]

persona_donut = go.Figure(go.Pie(
    labels=[f"{l.replace('-', ' ').title()}" for l in _labels],
    values=_values,
    hole=0.55,
    marker=dict(colors=_colors, line=dict(color=BG, width=3)),
    textinfo="label+percent",
    textfont=dict(color=FG, size=13),
    hovertemplate="<b>%{label}</b><br>N = %{value:,}<br>Share = %{percent}<extra></extra>",
    direction="clockwise",
    sort=False,
))
persona_donut.update_layout(
    title=dict(text="User Persona Distribution", font=dict(size=18, color=FG), x=0.5, xanchor="center"),
    annotations=[dict(
        text=f"<b>{_total_users:,}</b><br><span style='font-size:11px;color:{FG2}'>total users</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(color=FG, size=15), align="center",
    )],
    showlegend=True,
    legend=dict(
        orientation="v", x=1.02, y=0.5, xanchor="left",
        font=dict(color=FG, size=12),
    ),
    margin=dict(l=30, r=130, t=80, b=40),
    width=560, height=400,
)
persona_donut.show()
print(f"Chart 1 rendered — Persona Donut ({_total_users:,} users)")

# Persona distribution table
_persona_dist_rows = [
    {"Persona": p, "N": _pcounts.get(p, 0), "Share_%": round(_pcounts.get(p, 0) / _total_users * 100, 1)}
    for p in PERSONA_ORDER if p in _pcounts.index
]
_persona_dist_df = pd.DataFrame(_persona_dist_rows)
print("\n  Persona Distribution:")
print(_persona_dist_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — FUNNEL CONVERSION RATES BY PERSONA (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
_FUNNEL_STEPS = ["entry", "canvas", "build", "run", "habit", "ship"]
_STEP_FLAGS = {
    "entry":  lambda d: pd.Series(True, index=d.index),
    "canvas": lambda d: d["time_to_first_canvas"].notna(),
    "build":  lambda d: d["time_to_first_build"].notna(),
    "run":    lambda d: d["time_to_first_run"].notna(),
    "habit":  lambda d: d["runs_on_2plus_distinct_days_within_14d"],
    "ship":   lambda d: d["shipped"],
}
_STEP_PAIRS  = list(zip(_FUNNEL_STEPS[:-1], _FUNNEL_STEPS[1:]))

_funnel_conv_rows = []
for _p in PERSONA_ORDER:
    _sub = persona_df[persona_df["persona"] == _p].copy()
    for _sa, _sb in _STEP_PAIRS:
        _in_prev = _STEP_FLAGS[_sa](_sub)
        _in_next = _STEP_FLAGS[_sb](_sub)
        _n_prev  = int(_in_prev.sum())
        _n_reach = int((_in_prev & _in_next).sum())
        _conv    = _n_reach / _n_prev * 100 if _n_prev > 0 else 0.0
        _funnel_conv_rows.append({
            "persona": _p,
            "transition": f"{_sa}→{_sb}",
            "n_prev": _n_prev,
            "n_reach": _n_reach,
            "conv_pct": round(_conv, 1),
        })

funnel_conv_df = pd.DataFrame(_funnel_conv_rows)
_transitions   = [f"{a}→{b}" for a, b in _STEP_PAIRS]

funnel_conv_chart = go.Figure()
for _p in PERSONA_ORDER:
    _pdata = funnel_conv_df[funnel_conv_df["persona"] == _p]
    _pdata = _pdata.set_index("transition").reindex(_transitions)
    funnel_conv_chart.add_trace(go.Bar(
        name=_p.replace("-", " ").title(),
        x=_transitions,
        y=_pdata["conv_pct"].values,
        marker_color=PERSONA_COLORS[_p],
        marker_line=dict(width=0),
        text=[f"{v:.1f}%" if not np.isnan(v) else "" for v in _pdata["conv_pct"].values],
        textposition="outside",
        textfont=dict(color=FG, size=11),
        hovertemplate="<b>%{x}</b><br>%{name}<br>Conv: %{y:.1f}%<br>N prev: %{customdata[0]:,} → N reach: %{customdata[1]:,}<extra></extra>",
        customdata=np.column_stack([_pdata["n_prev"].values, _pdata["n_reach"].values]),
    ))

funnel_conv_chart.update_layout(
    title=dict(text="Funnel Conversion Rates by Persona", font=dict(size=18, color=FG), x=0.5, xanchor="center"),
    barmode="group",
    bargap=0.25,
    bargroupgap=0.05,
    xaxis=dict(title="Funnel Step Transition", tickfont=dict(color=FG2, size=12)),
    yaxis=dict(title="Conversion Rate (%)", ticksuffix="%", range=[0, 115]),
    legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", font=dict(color=FG, size=12)),
    margin=dict(l=60, r=30, t=80, b=90),
    width=820, height=480,
)
funnel_conv_chart.show()
print(f"Chart 2 rendered — Funnel Conversion by Persona")

# Funnel conversion table
print("\n  Funnel Conversion Rates by Persona:")
print(funnel_conv_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — TIME-DELTA TABLE: MEDIAN HOURS BETWEEN MILESTONES BY PERSONA
# ══════════════════════════════════════════════════════════════════════════════
_TIME_MAP = {
    ("entry",  "canvas"): ("entry",                   "time_to_first_canvas"),
    ("canvas", "build"):  ("time_to_first_canvas",    "time_to_first_build"),
    ("build",  "run"):    ("time_to_first_build",     "time_to_first_run"),
    ("run",    "habit"):  (None,                      None),   # no direct TS for habit
    ("habit",  "ship"):   (None,                      "time_to_first_ship"),
}

_delta_rows = []
for _p in PERSONA_ORDER:
    _sub = persona_df[persona_df["persona"] == _p].copy()
    for _sa, _sb in _STEP_PAIRS:
        _col_a, _col_b = _TIME_MAP[(_sa, _sb)]
        # compute delta
        if _col_b is None:
            _med_h = np.nan
        elif _col_a is None or _col_a == "entry":
            # delta from t0 (= 0) to _col_b
            _vals = _sub[_col_b].dropna()
            _med_h = float(np.median(_vals)) if len(_vals) >= 2 else np.nan
        else:
            _mask = _sub[_col_a].notna() & _sub[_col_b].notna()
            _vals = _sub.loc[_mask, _col_b] - _sub.loc[_mask, _col_a]
            _med_h = float(np.median(_vals)) if _mask.sum() >= 2 else np.nan

        # format
        if np.isnan(_med_h):
            _fmt = "—"
        elif _med_h < 1:
            _fmt = f"{_med_h*60:.0f} min"
        elif _med_h < 24:
            _fmt = f"{_med_h:.1f} h"
        else:
            _fmt = f"{_med_h/24:.1f} days"

        _delta_rows.append({
            "Persona":    _p.replace("-", " ").title(),
            "Transition": f"{_sa} → {_sb}",
            "Median Δ":   _fmt,
            "_med_h":     _med_h,
        })

_delta_df = pd.DataFrame(_delta_rows)

# Build Plotly table
_pivot = _delta_df.pivot(index="Persona", columns="Transition", values="Median Δ")
_pivot = _pivot.reindex([p.replace("-", " ").title() for p in PERSONA_ORDER])
_col_order = [f"{a} → {b}" for a, b in _STEP_PAIRS]
_pivot = _pivot.reindex(columns=_col_order)

_header_vals = ["Persona"] + _col_order
_cell_vals   = [_pivot.index.tolist()] + [_pivot[c].tolist() for c in _col_order]

milestone_time_table = go.Figure(go.Table(
    columnwidth=[130] + [150] * len(_col_order),
    header=dict(
        values=[f"<b>{v}</b>" for v in _header_vals],
        fill_color="#2a2a32",
        font=dict(color=FG, size=13),
        align="center",
        line_color=BG,
        height=36,
    ),
    cells=dict(
        values=_cell_vals,
        fill_color=[
            [PERSONA_COLORS[p] for p in PERSONA_ORDER],
        ] + [["#23232a"] * 3] * len(_col_order),
        font=dict(color=FG, size=13),
        align=["left"] + ["center"] * len(_col_order),
        line_color=BG,
        height=32,
    ),
))
milestone_time_table.update_layout(
    title=dict(
        text="Median Time Between Milestones by Persona (canvas→build→run→habit→ship)",
        font=dict(size=15, color=FG), x=0.5, xanchor="center",
    ),
    margin=dict(l=20, r=20, t=70, b=20),
    width=900, height=220,
)
milestone_time_table.show()
print(f"Chart 3 rendered — Milestone time-delta table")

# Milestone time delta table
_delta_md = _delta_df[["Persona", "Transition", "Median Δ"]].copy()
print("\n  Median Time Between Milestones by Persona:")
print(_delta_md.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — MARKOV TRANSITION HEATMAP (from sequence_analysis)
# ══════════════════════════════════════════════════════════════════════════════
_tmat = transition_matrix_df.copy()
_tokens = list(_tmat.columns)   # CANONICAL_TOKENS order

# Build custom colorscale: dark bg → indigo → purple → gold
_colorscale = [
    [0.0,  BG],
    [0.15, "#1b1049"],
    [0.45, COL_AGENT],       # purple
    [0.75, "#3a86ff"],       # blue
    [1.0,  "#ffd400"],       # yellow accent
]

_zmax = float(_tmat.values.max())

markov_heatmap = go.Figure(go.Heatmap(
    z=_tmat.values,
    x=_tokens,
    y=_tokens,
    colorscale=_colorscale,
    zmin=0,
    zmax=_zmax,
    text=[[f"{v:.2f}" if v > 0.005 else "" for v in row] for row in _tmat.values],
    texttemplate="%{text}",
    textfont=dict(size=9, color=FG),
    hovertemplate="<b>%{y} → %{x}</b><br>P = %{z:.4f}<extra></extra>",
    colorbar=dict(
        title=dict(text="P(transition)", font=dict(color=FG, size=11)),
        tickfont=dict(color=FG2, size=10),
        bgcolor=BG,
        bordercolor=BG,
        thickness=14,
        len=0.85,
    ),
    xgap=2,
    ygap=2,
))
markov_heatmap.update_layout(
    title=dict(
        text="Markov Transition Probability Matrix<br><sup>Row-normalized: from-token (y) → to-token (x)</sup>",
        font=dict(size=17, color=FG), x=0.5, xanchor="center",
    ),
    xaxis=dict(
        title="TO token", tickangle=-40,
        tickfont=dict(color=FG2, size=11),
        side="bottom",
    ),
    yaxis=dict(
        title="FROM token",
        tickfont=dict(color=FG2, size=11),
        autorange="reversed",
    ),
    margin=dict(l=120, r=80, t=100, b=100),
    width=820, height=660,
)
markov_heatmap.show()
print(f"Chart 4 rendered — Markov Transition Heatmap ({_tmat.shape[0]}x{_tmat.shape[1]})")

# Markov transition matrix
print("\n  Markov Transition Matrix (row-normalized, rounded to 4dp):")
_tmat_md = transition_matrix_df.round(4).reset_index()
print(_tmat_md.to_string(index=False))

print("\n✅  viz_agent_impact — all 4 charts rendered successfully")
