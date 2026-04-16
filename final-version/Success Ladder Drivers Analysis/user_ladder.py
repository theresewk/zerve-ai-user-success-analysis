
import pandas as pd
import numpy as np

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Column references ──────────────────────────────────────────────────────────
USER_COL      = "person_id"          # stable user identifier
EVENT_COL     = "event"
TS_COL        = "timestamp"
SESSION_COL   = "prop_$session_id"   # session identifier (web events)

# ── Event-name constants (verified from data exploration) ──────────────────────
SIGN_IN_EVENTS   = {"sign_in"}
# first meaningful canvas engagement
CANVAS_EVENTS    = {"canvas_open", "canvas_create"}
# first build action (user OR agent creating a block)
BUILD_EVENTS     = {"block_create", "agent_tool_call_create_block_tool"}
# run events per spec
RUN_EVENTS       = {
    "run_block", "run_all_blocks", "run_from_block",
    "run_upto_block", "agent_tool_call_run_block_tool"
}
SHIP_EVENTS      = {"app_publish", "api_deploy", "hosted_apps_deploy"}
SHIP_USE_EVENTS  = {"hosted_apps_open"}

# ── Load & parse ───────────────────────────────────────────────────────────────
print("Preparing event data…")
df = event_df[[USER_COL, EVENT_COL, TS_COL, SESSION_COL]].copy()
df["ts"] = pd.to_datetime(df[TS_COL], utc=True, errors="coerce")
df = df.dropna(subset=["ts", USER_COL])
df = df.sort_values([USER_COL, "ts"])

print(f"  Rows after parse : {len(df):,}")
print(f"  Unique users     : {df[USER_COL].nunique():,}")

# ── Anchor t0: first sign_in, else first observed event ───────────────────────
sign_in_df   = df[df[EVENT_COL].isin(SIGN_IN_EVENTS)].groupby(USER_COL)["ts"].min().rename("t0_signin")
first_event  = df.groupby(USER_COL)["ts"].min().rename("t0_first")
t0_df        = pd.concat([sign_in_df, first_event], axis=1)
t0_df["t0"] = t0_df["t0_signin"].fillna(t0_df["t0_first"])

# ── Join t0 back to events ─────────────────────────────────────────────────────
df = df.merge(t0_df[["t0"]], on=USER_COL, how="left")
df["hours_since_t0"] = (df["ts"] - df["t0"]).dt.total_seconds() / 3600
df["days_since_t0"]  = df["hours_since_t0"] / 24
df["date"]           = df["ts"].dt.normalize()
df["t0_date"]        = df["t0"].dt.normalize()

# ── ACTIVATE flags ─────────────────────────────────────────────────────────────
run_df = df[df[EVENT_COL].isin(RUN_EVENTS)].copy()
first_run = run_df.groupby(USER_COL)["ts"].min().rename("first_run_ts")

# run_within_24h: any run event within first 24 h of t0
run_24h = (
    run_df[run_df["hours_since_t0"] <= 24]
    .groupby(USER_COL).size().gt(0).rename("run_within_24h")
)
# run_within_7d
run_7d = (
    run_df[run_df["days_since_t0"] <= 7]
    .groupby(USER_COL).size().gt(0).rename("run_within_7d")
)

# run_within_first_session: first run session == first web session for user
run_first_session = (
    run_df.sort_values("ts")
    .groupby(USER_COL)
    .first()
    .reset_index()[[USER_COL, SESSION_COL]]
    .rename(columns={SESSION_COL: "first_run_session"})
)
first_web_event = (
    df[df[SESSION_COL].notna()]
    .sort_values("ts")
    .groupby(USER_COL)
    .first()
    .reset_index()[[USER_COL, SESSION_COL]]
    .rename(columns={SESSION_COL: "t0_session"})
)
session_merge = first_web_event.merge(run_first_session, on=USER_COL, how="outer")
session_merge["run_within_first_session"] = (
    session_merge["t0_session"].notna() &
    session_merge["first_run_session"].notna() &
    (session_merge["t0_session"] == session_merge["first_run_session"])
).fillna(False)
run_first_sess_flag = session_merge.set_index(USER_COL)["run_within_first_session"]

# ── HABIT flags (relative to first run) ───────────────────────────────────────
run_df = run_df.merge(first_run.reset_index(), on=USER_COL, how="left")
run_df["hrs_since_first_run"]  = (run_df["ts"] - run_df["first_run_ts"]).dt.total_seconds() / 3600
run_df["days_since_first_run"] = run_df["hrs_since_first_run"] / 24

habit_window = run_df[run_df["days_since_first_run"].between(0, 14)]
habit_days   = habit_window.groupby(USER_COL)["date"].nunique()
runs_2plus_days = habit_days.ge(2).rename("runs_on_2plus_distinct_days_within_14d")

habit_sessions = (
    habit_window[habit_window[SESSION_COL].notna()]
    .groupby(USER_COL)[SESSION_COL].nunique()
)
runs_3plus_sess = habit_sessions.ge(3).rename("runs_in_3plus_distinct_sessions_within_14d")

returned_7 = (
    run_df[run_df["days_since_first_run"] > 7]
    .groupby(USER_COL).size().gt(0).rename("returned_after_day7")
)

# ── SHIP flags (within 60d from t0) ───────────────────────────────────────────
ship_df      = df[df[EVENT_COL].isin(SHIP_EVENTS) & (df["days_since_t0"] <= 60)].copy()
shipped_flag = ship_df.groupby(USER_COL).size().gt(0).rename("shipped")

first_ship   = ship_df.groupby(USER_COL)["ts"].min().rename("first_ship_ts").reset_index()
use_df       = df[df[EVENT_COL].isin(SHIP_USE_EVENTS)].copy()
ship_use     = first_ship.merge(use_df[[USER_COL, "ts"]].rename(columns={"ts": "use_ts"}), on=USER_COL, how="left")
ship_use["hrs_use_after_ship"] = (ship_use["use_ts"] - ship_use["first_ship_ts"]).dt.total_seconds() / 3600
ship_then_use = (
    ship_use[(ship_use["hrs_use_after_ship"] >= 0) & (ship_use["hrs_use_after_ship"] <= 168)]
    .groupby(USER_COL).size().gt(0).rename("ship_then_use")
)

# ── TIME-TO metrics (hours from t0) ───────────────────────────────────────────
canvas_df = df[df[EVENT_COL].isin(CANVAS_EVENTS)]
time_to_canvas = canvas_df.groupby(USER_COL)["hours_since_t0"].min().rename("time_to_first_canvas")

build_df = df[df[EVENT_COL].isin(BUILD_EVENTS)]
time_to_build = build_df.groupby(USER_COL)["hours_since_t0"].min().rename("time_to_first_build")

time_to_run = run_df.groupby(USER_COL)["hours_since_t0"].min().rename("time_to_first_run")

time_to_ship = ship_df.groupby(USER_COL)["hours_since_t0"].min().rename("time_to_first_ship")

# ── Assemble user_ladder_df ────────────────────────────────────────────────────
all_users = df[USER_COL].unique()
user_ladder_df = pd.DataFrame({USER_COL: all_users}).set_index(USER_COL)

bool_flags = {
    "run_within_24h":                              run_24h,
    "run_within_7d":                               run_7d,
    "run_within_first_session":                    run_first_sess_flag,
    "runs_on_2plus_distinct_days_within_14d":      runs_2plus_days,
    "runs_in_3plus_distinct_sessions_within_14d":  runs_3plus_sess,
    "returned_after_day7":                         returned_7,
    "shipped":                                     shipped_flag,
    "ship_then_use":                               ship_then_use,
}
for _col_name, _series in bool_flags.items():
    user_ladder_df[_col_name] = _series.reindex(user_ladder_df.index).fillna(False).astype(bool)

time_col_map = {
    "time_to_first_canvas": time_to_canvas,
    "time_to_first_build":  time_to_build,
    "time_to_first_run":    time_to_run,
    "time_to_first_ship":   time_to_ship,
}
for _col_name, _series in time_col_map.items():
    user_ladder_df[_col_name] = _series.reindex(user_ladder_df.index)

user_ladder_df = user_ladder_df.reset_index()

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  user_ladder_df  →  shape: {user_ladder_df.shape}")
print(f"  Unique users    : {user_ladder_df[USER_COL].nunique():,}")
print(f"{'='*60}")

flag_list = list(bool_flags.keys())
summary_rows = []
for _fc in flag_list:
    _n_true = int(user_ladder_df[_fc].sum())
    _rate   = _n_true / len(user_ladder_df) * 100
    summary_rows.append({"Flag": _fc, "True Count": _n_true, "Rate %": round(_rate, 2)})

ladder_summary = pd.DataFrame(summary_rows)

print("\n  SUCCESS LADDER FLAG RATES\n")
# Format as a clean table without requiring tabulate
_col_widths = {
    "Flag":        max(len("Flag"),        ladder_summary["Flag"].str.len().max()),
    "True Count":  max(len("True Count"),  ladder_summary["True Count"].astype(str).str.len().max()),
    "Rate %":      max(len("Rate %"),      ladder_summary["Rate %"].astype(str).str.len().max()),
}
_header = (
    f"  {'Flag':<{_col_widths['Flag']}}  "
    f"{'True Count':>{_col_widths['True Count']}}  "
    f"{'Rate %':>{_col_widths['Rate %']}}"
)
_sep = "  " + "-" * (_col_widths["Flag"] + _col_widths["True Count"] + _col_widths["Rate %"] + 6)
print(_header)
print(_sep)
for _, _row in ladder_summary.iterrows():
    print(
        f"  {_row['Flag']:<{_col_widths['Flag']}}  "
        f"{_row['True Count']:>{_col_widths['True Count']}}  "
        f"{_row['Rate %']:>{_col_widths['Rate %']}}"
    )

# ── Time-to metrics summary ────────────────────────────────────────────────────
print(f"\n  TIME-TO METRICS (hours from t0)  — medians & coverage")
_time_summary_rows = []
for _tc in time_col_map:
    _med = user_ladder_df[_tc].median()
    _cnt = user_ladder_df[_tc].notna().sum()
    _time_summary_rows.append({"Metric": _tc, "Median (h)": round(_med, 2), "N (reached)": _cnt})
_time_summary_df = pd.DataFrame(_time_summary_rows)

_tw = {
    "Metric":      max(len("Metric"),      _time_summary_df["Metric"].str.len().max()),
    "Median (h)":  max(len("Median (h)"),  _time_summary_df["Median (h)"].astype(str).str.len().max()),
    "N (reached)": max(len("N (reached)"), _time_summary_df["N (reached)"].astype(str).str.len().max()),
}
_th = (
    f"  {'Metric':<{_tw['Metric']}}  "
    f"{'Median (h)':>{_tw['Median (h)']}}  "
    f"{'N (reached)':>{_tw['N (reached)']}}"
)
_ts = "  " + "-" * (_tw["Metric"] + _tw["Median (h)"] + _tw["N (reached)"] + 6)
print(_th)
print(_ts)
for _, _row in _time_summary_df.iterrows():
    print(
        f"  {_row['Metric']:<{_tw['Metric']}}  "
        f"{_row['Median (h)']:>{_tw['Median (h)']}}  "
        f"{_row['N (reached)']:>{_tw['N (reached)']}}"
    )
