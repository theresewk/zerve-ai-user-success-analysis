
import pandas as pd
import numpy as np

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Column constants ───────────────────────────────────────────────────────────
_USER  = "person_id"
_EVENT = "event"
_TS    = "timestamp"

# ── Add-credits intent events ──────────────────────────────────────────────────
_ADD_CREDITS_INTENT_EVENTS = {
    "add_credits",
    "clicked_add_credits",
    "ai_credit_banner_clicked",
    "agent_add_credits_button_clicked",
    "add_credits_click",
    "billing_open",
    "billing_clicked",
    "upgrade_clicked",
    "upgrade_click",
    "purchase_credits",
}

# ── Run / Stop events ──────────────────────────────────────────────────────────
_RUN_EVENTS = {
    "run_block", "run_all_blocks", "run_from_block",
    "run_upto_block", "agent_tool_call_run_block_tool",
}
_STOP_EVENTS = {
    "stop_block", "stop_all_blocks",
    "cancel_run", "stop_run", "block_stopped",
    "agent_tool_call_stop_block", "cancel_block",
}

# ── Agent error events ─────────────────────────────────────────────────────────
_AGENT_ERROR_ASSIST_EVENTS = {"agent_open_error_assist"}
_AGENT_RETRY_EVENTS        = {"agent_retry_message_button_clicked"}

# ── Parse events ───────────────────────────────────────────────────────────────
print("Parsing events for friction feature engineering…")
_raw = event_df[[_USER, _EVENT, _TS]].copy()
_raw["ts"] = pd.to_datetime(_raw[_TS], utc=True, errors="coerce")
_raw = _raw.dropna(subset=["ts", _USER]).sort_values([_USER, "ts"])
print(f"  Events parsed: {len(_raw):,}  |  Users: {_raw[_USER].nunique():,}")

# Confirm key friction events exist in raw data
print(f"  credits_exceeded events:  {_raw[_EVENT].str.contains('credits_exceeded', case=False, na=False).sum():,}")
print(f"  credits_below events:     {_raw[_EVENT].str.contains('credits_below', case=False, na=False).sum():,}")
print(f"  stop_block events:        {(_raw[_EVENT] == 'stop_block').sum():,}")

# ── Attach t0 ─────────────────────────────────────────────────────────────────
_t0 = t0_df[["t0"]].copy()
_raw = _raw.merge(_t0, on=_USER, how="left")
_raw["hours_since_t0"] = (_raw["ts"] - _raw["t0"]).dt.total_seconds() / 3600

print(f"  events 0-24h from t0 : {_raw['hours_since_t0'].between(0,24).sum():,}")
print(f"  events 0-7d from t0  : {_raw['hours_since_t0'].between(0,168).sum():,}")

# ── Base user list (fixed order, 4,771 users) ─────────────────────────────────
_base_users = user_ladder_df[_USER].tolist()
_N          = len(_base_users)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cnt(wdf, mask):
    """Per-user event count for events where mask is True; aligned to _base_users."""
    return wdf[mask].groupby(_USER).size().reindex(_base_users).fillna(0).astype(int)


def _fts(wdf, mask):
    """Per-user first timestamp; aligned to _base_users with NaT for absent users."""
    return wdf[mask].groupby(_USER)["ts"].min().reindex(_base_users)


def _ts_delta_hours(ts_a, ts_b):
    """
    Compute (ts_a - ts_b) in hours as a float Series.
    Both ts_a and ts_b are Series indexed by _base_users, dtype datetime64[ns, UTC].
    Uses nanoseconds directly to avoid numpy.timedelta64.total_seconds issues.
    """
    _a_ns = ts_a.values.view("int64").astype(float)
    _b_ns = ts_b.values.view("int64").astype(float)
    _iNaT = np.datetime64("NaT").view("int64")
    _delta = _a_ns - _b_ns
    _delta[(_a_ns == _iNaT) | (_b_ns == _iNaT)] = np.nan
    return pd.Series(_delta / 3.6e12, index=_base_users)  # ns → hours


def _before_first(first_ts_a, first_ts_b):
    """
    Flag: ts_a is set AND (ts_b is NaT OR ts_a < ts_b).
    Both series are aligned to _base_users.
    Returns bool Series.
    """
    _has_a  = first_ts_a.notna()
    _no_b   = first_ts_b.isna()
    _a_ns = first_ts_a.values.view("int64").astype(float)
    _b_ns = first_ts_b.values.view("int64").astype(float)
    _iNaT = float(np.datetime64("NaT").view("int64"))
    _a_lt_b = pd.Series(
        np.where((_a_ns == _iNaT) | (_b_ns == _iNaT), False, _a_ns < _b_ns),
        index=_base_users
    )
    return (_has_a & (_no_b | _a_lt_b)).fillna(False).astype(bool)


def compute_window_features(window_h):
    sfx = "24h" if window_h <= 25 else "7d"
    _w = _raw[_raw["hours_since_t0"].between(0, window_h, inclusive="both")].copy()
    print(f"\n  [{sfx}] window: {len(_w):,} events, {_w[_USER].nunique():,} active users")

    # ── 1. CREDITS FRICTION ──────────────────────────────────────────────────
    _warn_m  = _w[_EVENT].str.contains(
        r"credits_below|credits_warning|credit_warning|low_credits|ai_credits_low|ai_credits_critical",
        case=False, na=False, regex=True
    )
    _exc_m   = _w[_EVENT].str.contains(
        r"credits_exceeded|out_of_credits|credit_exceeded|credit_limit_reached",
        case=False, na=False, regex=True
    )
    _intent_m = _w[_EVENT].isin(_ADD_CREDITS_INTENT_EVENTS)
    print(f"    credits warn={_warn_m.sum():,}  exceeded={_exc_m.sum():,}  intent={_intent_m.sum():,}")

    _warn_count   = _cnt(_w, _warn_m).rename(f"credits_warn_count_{sfx}")
    _exc_count    = _cnt(_w, _exc_m).rename(f"credits_exceeded_count_{sfx}")
    _intent_count = _cnt(_w, _intent_m).rename(f"credits_add_intent_count_{sfx}")

    _fric_m      = _warn_m | _exc_m
    _run_m       = _w[_EVENT].isin(_RUN_EVENTS)
    _first_fric  = _fts(_w, _fric_m)
    _first_run_w = _fts(_w, _run_m)

    _cred_before_run     = _before_first(_first_fric, _first_run_w)
    _cred_before_run.name = f"credits_friction_before_run_{sfx}"

    _first_intent = _fts(_w, _intent_m)
    _delta_h      = _ts_delta_hours(_first_intent, _first_fric)
    _delta_h.name = f"credits_warn_to_intent_hours_{sfx}"

    _cohort = pd.Series("no_friction", index=_base_users, dtype=object)
    _cohort[_warn_count > 0] = "warn_only"
    _cohort[_exc_count  > 0] = "exceeded"
    _cohort.name = f"credits_cohort_{sfx}"

    # ── 2. EXECUTION DISRUPTION ──────────────────────────────────────────────
    _stop_m    = _w[_EVENT].isin(_STOP_EVENTS)
    print(f"    stop={_stop_m.sum():,}  run={_run_m.sum():,}")
    _stop_count = _cnt(_w, _stop_m).rename(f"stop_count_{sfx}")

    _run_ev  = _w[_run_m][[_USER, "ts"]].rename(columns={"ts": "run_ts"}).reset_index(drop=True)
    _stop_ev = _w[_stop_m][[_USER, "ts"]].rename(columns={"ts": "stop_ts"}).reset_index(drop=True)
    _sar = pd.Series(np.nan, index=_base_users, name=f"stop_after_run_rate_{sfx}")
    if len(_run_ev) > 0 and len(_stop_ev) > 0:
        _rs = _run_ev.merge(_stop_ev, on=_USER, how="inner")
        _rs["_gap_s"] = (_rs["stop_ts"] - _rs["run_ts"]).dt.total_seconds()
        _r60 = _rs[_rs["_gap_s"].between(0, 60)]
        _spr = _r60.groupby(_USER).size()
        _tot = _run_ev.groupby(_USER).size()
        _sar = (_spr / _tot).reindex(_base_users).rename(f"stop_after_run_rate_{sfx}")

    _first_stop     = _fts(_w, _stop_m)
    _stop_before_run = _before_first(_first_stop, _first_run_w)
    _stop_before_run.name = f"stop_before_first_run_{sfx}"

    # ── 3. OPERATIONAL BLOCKERS ──────────────────────────────────────────────
    _seats_m = _w[_EVENT].str.contains(
        r"seats_exceeded|seat_limit|seats_limit", case=False, na=False, regex=True
    )
    print(f"    seats={_seats_m.sum():,}")
    _seats_count = _cnt(_w, _seats_m).rename(f"seats_exceeded_count_{sfx}")
    _seats_flag  = (_seats_count > 0).rename(f"any_seats_friction_{sfx}")

    # ── 4. AGENT ERROR SIGNALS ────────────────────────────────────────────────
    _err_m   = _w[_EVENT].isin(_AGENT_ERROR_ASSIST_EVENTS)
    _retry_m = _w[_EVENT].isin(_AGENT_RETRY_EVENTS)
    print(f"    agent_error_assist={_err_m.sum():,}  retry={_retry_m.sum():,}")
    _err_count   = _cnt(_w, _err_m).rename(f"agent_error_assist_count_{sfx}")
    _retry_count = _cnt(_w, _retry_m).rename(f"agent_retry_count_{sfx}")
    _agent_err   = ((_err_count > 0) | (_retry_count > 0)).rename(f"any_agent_error_friction_{sfx}")

    return pd.concat([
        _warn_count, _exc_count, _intent_count,
        _cred_before_run, _delta_h, _cohort,
        _stop_count, _sar, _stop_before_run,
        _seats_count, _seats_flag,
        _err_count, _retry_count, _agent_err,
    ], axis=1)


# ── Compute for both windows ───────────────────────────────────────────────────
print("\n" + "="*60)
print("Computing 24h window friction features…")
_feats_24h = compute_window_features(window_h=24)

print("\n" + "="*60)
print("Computing 7d window friction features…")
_feats_7d = compute_window_features(window_h=168)

# ── Assemble friction_features_df (one row per user) ─────────────────────────
_f24 = _feats_24h.reset_index().rename(columns={"index": _USER})
_f7d = _feats_7d.reset_index().rename(columns={"index": _USER})

friction_features_df = (
    pd.DataFrame({_USER: _base_users})
    .merge(_f24, on=_USER, how="left")
    .merge(_f7d,  on=_USER, how="left")
)

# Join persona
_persona_join = persona_df[[_USER, "persona"]].drop_duplicates(subset=[_USER])
friction_features_df = friction_features_df.merge(_persona_join, on=_USER, how="left")

# ── Sanity checks ──────────────────────────────────────────────────────────────
assert friction_features_df[_USER].nunique() == _N
assert len(friction_features_df) == _N

print(f"\n{'='*70}")
print(f"  friction_features_df shape : {friction_features_df.shape}")
print(f"  Unique users               : {friction_features_df[_USER].nunique():,}")
print(f"  Feature columns            : {friction_features_df.shape[1] - 2}")
print(f"{'='*70}")

# ── NON-ZERO RATE SUMMARY ─────────────────────────────────────────────────────
print(f"\n{'═'*82}")
print(f"  FRICTION FEATURE NON-ZERO / HIT RATES  (N = {_N:,})")
print(f"{'═'*82}")

_feature_cols = [c for c in friction_features_df.columns if c not in (_USER, "persona")]
_rate_rows = []
for _col in _feature_cols:
    _s  = friction_features_df[_col]
    _dt = str(_s.dtype)[:6]
    if _s.dtype == bool:
        _nh = int(_s.sum())
    elif _s.dtype == object:
        if "cohort" in _col:
            _nh = int((_s != "no_friction").sum())
        else:
            _nh = int(_s.notna().sum())
    else:
        if "rate" in _col or "hours" in _col:
            _nh = int(_s.notna().sum())
        else:
            _nh = int((_s > 0).sum())
    _rt = _nh / _N * 100
    _rate_rows.append({"Feature": _col, "Dtype": _dt, "N_hit": _nh, "Rate_%": round(_rt, 2)})

_rate_summary_df = pd.DataFrame(_rate_rows)
print(_rate_summary_df.to_string(index=False))

# ── Credits cohort breakdown ──────────────────────────────────────────────────
for _sfx in ["24h", "7d"]:
    _col  = f"credits_cohort_{_sfx}"
    _dist = friction_features_df[_col].value_counts()
    _dist_df = _dist.reset_index()
    _dist_df.columns = ["cohort", "count"]
    _dist_df["pct_%"] = (_dist_df["count"] / _N * 100).round(2)
    print(f"\n  Credits cohort ({_sfx}):")
    print(_dist_df.to_string(index=False))

print(f"\n{'═'*82}")
print("  ✅  friction_features_df  COMPLETE")
print(f"  Shape: {friction_features_df.shape[0]:,} rows × {friction_features_df.shape[1]} cols")
print(f"{'═'*82}")
