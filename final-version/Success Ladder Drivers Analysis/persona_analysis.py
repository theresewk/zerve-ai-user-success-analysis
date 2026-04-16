
import pandas as pd
import numpy as np

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Constants (re-declare for isolation) ──────────────────────────────────────
USER_COL    = "person_id"
EVENT_COL   = "event"
TS_COL      = "timestamp"

AGENT_EVENTS = {
    "agent_tool_call_create_block_tool",
    "agent_tool_call_run_block_tool",
    "agent_tool_call_refactor_block_tool",
    "agent_tool_call_delete_block_tool",
    "agent_tool_call_create_edges_tool",
    "agent_message_sent",
    "agent_run_started",
    "agent_run_completed",
    "agent_run_failed",
}

BUILD_EVENTS = {
    "block_create",
    "agent_tool_call_create_block_tool",
}

RUN_EVENTS = {
    "run_block", "run_all_blocks", "run_from_block",
    "run_upto_block", "agent_tool_call_run_block_tool",
}

CANVAS_EVENTS = {"canvas_open", "canvas_create"}
SHIP_EVENTS   = {"app_publish", "api_deploy", "hosted_apps_deploy"}

# ── Parse events ──────────────────────────────────────────────────────────────
print("Preparing event data for persona assignment...")
_ev = event_df[[USER_COL, EVENT_COL, TS_COL]].copy()
_ev["ts"] = pd.to_datetime(_ev[TS_COL], utc=True, errors="coerce")
_ev = _ev.dropna(subset=["ts", USER_COL]).sort_values([USER_COL, "ts"])

# Bring in t0
_t0 = t0_df[["t0"]].copy()
_ev = _ev.merge(_t0, on=USER_COL, how="left")
_ev["hours_since_t0"] = (_ev["ts"] - _ev["t0"]).dt.total_seconds() / 3600
_ev["days_since_t0"]  = _ev["hours_since_t0"] / 24

# ── PERSONA RULES ─────────────────────────────────────────────────────────────
_agent_24h = (
    _ev[_ev[EVENT_COL].isin(AGENT_EVENTS) & (_ev["hours_since_t0"] <= 24)]
    .groupby(USER_COL).size().gt(0)
)

_build_7d = (
    _ev[_ev[EVENT_COL].isin(BUILD_EVENTS) & (_ev["days_since_t0"] <= 7)]
    .groupby(USER_COL).size().gt(0)
)
_run_7d_flag = (
    _ev[_ev[EVENT_COL].isin(RUN_EVENTS) & (_ev["days_since_t0"] <= 7)]
    .groupby(USER_COL).size().gt(0)
)

_all = user_ladder_df[["person_id"]].copy()
_all["_agent_24h"]  = _all["person_id"].map(_agent_24h).fillna(False)
_all["_build_7d"]   = _all["person_id"].map(_build_7d).fillna(False)
_all["_run_7d"]     = _all["person_id"].map(_run_7d_flag).fillna(False)

def _assign_persona(row):
    if row["_agent_24h"]:
        return "agent-led"
    elif row["_build_7d"] or row["_run_7d"]:
        return "manual-led"
    else:
        return "observer-only"

persona_df = _all.assign(persona=_all.apply(_assign_persona, axis=1))
persona_df = persona_df.merge(
    user_ladder_df.drop(columns=[c for c in user_ladder_df.columns if c == "_agent_24h"], errors="ignore"),
    on="person_id", how="left"
)
persona_df = persona_df.drop(columns=["_agent_24h", "_build_7d", "_run_7d"])

print(f"  persona_df shape: {persona_df.shape}")
print(f"  Persona column: {persona_df['persona'].value_counts().to_dict()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — PERSONA SIZE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
SEP = "═" * 80
print(f"\n{SEP}")
print("  TABLE 1 — PERSONA SIZE DISTRIBUTION")
print(SEP)
total = len(persona_df)
persona_counts = persona_df["persona"].value_counts()
_t1_rows = []
for persona_name, count in persona_counts.items():
    pct = count / total * 100
    _t1_rows.append({"Persona": persona_name, "N_Users": count, "Pct_%": round(pct, 1)})
_t1_rows.append({"Persona": "TOTAL", "N_Users": total, "Pct_%": 100.0})
_t1_df = pd.DataFrame(_t1_rows)
print(_t1_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — LADDER FLAG RATES BY PERSONA
# ─────────────────────────────────────────────────────────────────────────────
PERSONA_FLAGS = [
    "run_within_24h",
    "run_within_7d",
    "run_within_first_session",
    "runs_on_2plus_distinct_days_within_14d",
    "runs_in_3plus_distinct_sessions_within_14d",
    "returned_after_day7",
    "shipped",
    "ship_then_use",
]
SHORT_FLAGS = [
    "run_24h", "run_7d", "run_1sess", "2d_14d", "3s_14d", "ret_d7", "shpd", "ship_use"
]

print(f"\n{SEP}")
print("  TABLE 2 — LADDER FLAG RATES BY PERSONA")
print(SEP)

PERSONA_ORDER = ["agent-led", "manual-led", "observer-only"]
_t2_rows = []
for persona_name in PERSONA_ORDER:
    grp = persona_df[persona_df["persona"] == persona_name]
    n   = len(grp)
    row = {"Persona": persona_name, "N": n}
    for f, sf in zip(PERSONA_FLAGS, SHORT_FLAGS):
        row[sf] = round(grp[f].mean() * 100, 1)
    _t2_rows.append(row)
_t2_df = pd.DataFrame(_t2_rows)
print(_t2_df.to_string(index=False))
print(f"\n  Flag key: {', '.join(f'{s}={f}' for s, f in zip(SHORT_FLAGS, PERSONA_FLAGS))}")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — STEP FUNNEL: CONVERSION RATES + MEDIAN TIME BETWEEN STEPS
# ─────────────────────────────────────────────────────────────────────────────
FUNNEL_STEPS = ["entry", "canvas", "build", "run", "habit", "ship"]

def _funnel_flags(sub_df):
    return {
        "entry":  pd.Series(True,  index=sub_df.index),
        "canvas": sub_df["time_to_first_canvas"].notna(),
        "build":  sub_df["time_to_first_build"].notna(),
        "run":    sub_df["time_to_first_run"].notna(),
        "habit":  sub_df["runs_on_2plus_distinct_days_within_14d"],
        "ship":   sub_df["shipped"],
    }

def _median_time_between(sub_df, step_a, step_b):
    TIME_MAP = {
        "entry":  None,
        "canvas": "time_to_first_canvas",
        "build":  "time_to_first_build",
        "run":    "time_to_first_run",
        "habit":  None,
        "ship":   "time_to_first_ship",
    }
    col_a = TIME_MAP[step_a]
    col_b = TIME_MAP[step_b]

    if col_b is None:
        return np.nan

    if col_a is None:
        mask = sub_df[col_b].notna()
        vals = sub_df.loc[mask, col_b]
    else:
        mask = sub_df[col_a].notna() & sub_df[col_b].notna()
        vals = sub_df.loc[mask, col_b] - sub_df.loc[mask, col_a]

    if len(vals) < 2:
        return np.nan
    return float(np.median(vals))


print(f"\n{SEP}")
print("  TABLE 3 — STEP FUNNEL: CONVERSION RATES + MEDIAN HOURS BETWEEN STEPS, BY PERSONA")
print(SEP)
print("  Steps: entry → canvas → build → run → habit → ship")
print(f"  Conversion = % of PREVIOUS step who reached NEXT step")
print(f"  Med_Δh     = median hours from previous step to current step\n")

step_pairs = list(zip(FUNNEL_STEPS[:-1], FUNNEL_STEPS[1:]))

_t3_rows = []
for persona_name in PERSONA_ORDER:
    grp = persona_df[persona_df["persona"] == persona_name].copy()
    flags = _funnel_flags(grp)

    for step_a, step_b in step_pairs:
        in_prev  = flags[step_a]
        in_next  = flags[step_b]
        n_prev   = int(in_prev.sum())
        n_reach  = int((in_prev & in_next).sum())
        conv_pct = n_reach / n_prev * 100 if n_prev > 0 else 0.0

        sub = grp[in_prev]
        med_h = _median_time_between(sub, step_a, step_b)
        med_str = f"{med_h:.1f}" if not np.isnan(med_h) else "n/a"

        _t3_rows.append({
            "Persona": persona_name,
            "Transition": f"{step_a}→{step_b}",
            "N_prev": n_prev,
            "N_reach": n_reach,
            "Conv_%": round(conv_pct, 1),
            "Med_Δh": med_str,
        })

_t3_df = pd.DataFrame(_t3_rows)
print(_t3_df.to_string(index=False))

print(f"\n{SEP}")
print("  ✅  PERSONA ANALYSIS COMPLETE")
print(f"  persona_df exported — {len(persona_df):,} rows × {persona_df.shape[1]} cols")
print(f"  Persona groups: {dict(persona_df['persona'].value_counts())}")
print(SEP)
