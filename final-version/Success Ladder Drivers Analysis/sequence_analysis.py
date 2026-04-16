
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

# ── Pandas display settings ────────────────────────────────────────────────────
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ── Zerve design system colors ─────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# 1. CANONICAL TOKEN MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def map_to_token(event_name):
    """Map a raw event name to a canonical alphabet token."""
    e = str(event_name).lower().strip()

    if e in ("sign_in", "sign_out", "sign_up", "signup", "login", "logout"):
        return "AUTH"

    if e in ("canvas_open", "canvas_view", "canvas_create", "layer_open",
             "canvas_viewed", "canvas_loaded"):
        return "CANVAS"

    if e in ("block_create", "block_edit", "block_update", "block_updated",
             "edge_create", "edge_delete", "block_delete", "block_created",
             "agent_tool_call_create_block_tool",
             "agent_tool_call_refactor_block_tool",
             "agent_tool_call_delete_block_tool",
             "agent_tool_call_create_edges_tool"):
        return "BUILD"

    if e in ("run_block", "run_all_blocks", "run_from_block", "run_upto_block",
             "agent_tool_call_run_block_tool"):
        return "RUN"

    if e.startswith("agent_tool_call_") or e.startswith("agent_"):
        return "AGENT_TOOL"

    if "rerun" in e or "re_run" in e or "iterate" in e:
        return "ITERATE"

    if "credits_warning" in e or "low_credits" in e or "credit_warning" in e:
        return "CREDITS_WARN"

    if "credits_exceeded" in e or "out_of_credits" in e or "credit_exceeded" in e:
        return "CREDITS_EXCEEDED"

    if e in ("add_credits_click", "billing_open", "add_credits", "billing_clicked",
             "upgrade_clicked", "upgrade_click", "purchase_credits"):
        return "ADD_CREDITS_INTENT"

    if e in ("stop_block", "cancel_run", "stop_run", "block_stopped",
             "agent_tool_call_stop_block", "cancel_block"):
        return "STOP"

    if e in ("app_publish", "api_deploy", "hosted_apps_deploy",
             "app_deployed", "deploy", "publish_app", "publish"):
        return "SHIP"

    if e == "share_canvas" or e.startswith("invite_") or "share" in e:
        return "SHARE"

    return "OTHER"


# ── Apply token mapping ────────────────────────────────────────────────────────
print("Mapping events to canonical tokens...")
_seq_ev = event_df[["person_id", "event", "timestamp", "prop_$session_id"]].copy()
_seq_ev["ts"] = pd.to_datetime(_seq_ev["timestamp"], utc=True, errors="coerce")
_seq_ev = _seq_ev.dropna(subset=["ts", "person_id"]).sort_values(["person_id", "ts"])
_seq_ev["token"] = _seq_ev["event"].apply(map_to_token)

# Token distribution
_token_counts = _seq_ev["token"].value_counts()
_token_dist_df = _token_counts.reset_index()
_token_dist_df.columns = ["token", "count"]
_token_dist_df["pct_%"] = (_token_dist_df["count"] / len(_seq_ev) * 100).round(1)
print("\n  Token distribution (all events):")
print(_token_dist_df.to_string(index=False))

CANONICAL_TOKENS = ["AUTH", "CANVAS", "BUILD", "RUN", "AGENT_TOOL", "ITERATE",
                    "CREDITS_WARN", "CREDITS_EXCEEDED", "ADD_CREDITS_INTENT",
                    "STOP", "SHIP", "SHARE", "OTHER"]
_found = set(_seq_ev["token"].unique())
_missing = [t for t in CANONICAL_TOKENS if t not in _found]
print(f"\n  Canonical tokens found: {sorted(_found)}")
if _missing:
    print(f"  Not seen in data (no events map there): {_missing}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SESSIONIZE with 30-min inactivity gap
# ══════════════════════════════════════════════════════════════════════════════
print("\nSessionizing with 30-min inactivity gap...")
_seq_ev = _seq_ev.sort_values(["person_id", "ts"])
_seq_ev["_prev_ts"]  = _seq_ev.groupby("person_id")["ts"].shift(1)
_seq_ev["_gap_min"]  = (_seq_ev["ts"] - _seq_ev["_prev_ts"]).dt.total_seconds() / 60
_seq_ev["_new_sess"] = (
    (_seq_ev["person_id"] != _seq_ev["person_id"].shift(1)) |
    (_seq_ev["_gap_min"] > 30) |
    (_seq_ev["_gap_min"].isna())
)
_seq_ev["session_id"] = _seq_ev["_new_sess"].cumsum()

n_sessions = _seq_ev["session_id"].nunique()
n_users    = _seq_ev["person_id"].nunique()
print(f"  Sessions created : {n_sessions:,}")
print(f"  Users            : {n_users:,}")
print(f"  Avg sessions/user: {n_sessions/n_users:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD SESSION SEQUENCES (collapse consecutive duplicates)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding session sequences (collapsing consecutive duplicates)...")

def collapse_consecutive(tokens):
    if len(tokens) == 0:
        return []
    result = [tokens[0]]
    for t in tokens[1:]:
        if t != result[-1]:
            result.append(t)
    return result

_sess_seqs = (
    _seq_ev.groupby(["session_id", "person_id"])["token"]
    .apply(list)
    .reset_index()
)
_sess_seqs.columns = ["session_id", "person_id", "token_list"]
_sess_seqs["seq_collapsed"] = _sess_seqs["token_list"].apply(collapse_consecutive)
_seq_sep = "\u2192"  # → arrow
_sess_seqs["seq_str"] = _sess_seqs["seq_collapsed"].apply(lambda x: _seq_sep.join(x))
_sess_seqs["seq_len"] = _sess_seqs["seq_collapsed"].apply(len)

_all_seqs = _sess_seqs.copy()
print(f"  Total sessions: {len(_all_seqs):,}")
print(f"  Median seq length (collapsed): {_all_seqs['seq_len'].median():.1f}")

# Join persona
_all_seqs = _all_seqs.merge(
    persona_df[["person_id", "persona"]].drop_duplicates(),
    on="person_id", how="left"
)
_all_seqs["persona"] = _all_seqs["persona"].fillna("unknown")
total_sessions = len(_all_seqs)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TOP 20 SESSION SEQUENCES OVERALL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  TOP 20 SESSION SEQUENCES OVERALL")
print("="*80)

_overall_counts = Counter(_all_seqs["seq_str"])
_top20_overall  = _overall_counts.most_common(20)

session_sequences_rows = []
_top20_df_rows = []
for _rank, (_seq, _cnt) in enumerate(_top20_overall, 1):
    _pct = _cnt / total_sessions * 100
    _top20_df_rows.append({"rank": _rank, "count": _cnt, "pct_sessions_%": round(_pct, 2), "sequence": _seq})
    session_sequences_rows.append({
        "rank": _rank, "sequence": _seq, "count": _cnt,
        "pct_sessions": round(_pct, 4), "scope": "OVERALL"
    })

_top20_df = pd.DataFrame(_top20_df_rows)
print(_top20_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 5. TOP 10 SESSION SEQUENCES PER PERSONA
# ══════════════════════════════════════════════════════════════════════════════
_PERSONA_ORDER = ["agent-led", "manual-led", "observer-only"]

for _persona in _PERSONA_ORDER:
    _psess   = _all_seqs[_all_seqs["persona"] == _persona]
    _n_p     = len(_psess)
    _pcounts = Counter(_psess["seq_str"])
    _top10_p = _pcounts.most_common(10)

    print(f"\n{'='*80}")
    print(f"  TOP 10 SEQUENCES — Persona: {_persona.upper()}  (N sessions = {_n_p:,})")
    print("="*80)

    _p10_rows = []
    for _rank, (_seq, _cnt) in enumerate(_top10_p, 1):
        _pct = _cnt / _n_p * 100 if _n_p > 0 else 0
        _p10_rows.append({"rank": _rank, "count": _cnt, "pct_sessions_%": round(_pct, 1), "sequence": _seq})
        session_sequences_rows.append({
            "rank": _rank, "sequence": _seq, "count": _cnt,
            "pct_sessions": round(_pct, 4), "scope": "PERSONA:" + _persona
        })
    if _p10_rows:
        print(pd.DataFrame(_p10_rows).to_string(index=False))

session_sequences_df = pd.DataFrame(session_sequences_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKOV TRANSITION PROBABILITY MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  MARKOV TRANSITION PROBABILITY MATRIX")
print("="*80)

_transition_counts = Counter()
for _seq_list in _all_seqs["seq_collapsed"]:
    for _i in range(len(_seq_list) - 1):
        _transition_counts[(_seq_list[_i], _seq_list[_i+1])] += 1

_n_tok  = len(CANONICAL_TOKENS)
_tok_idx = {t: i for i, t in enumerate(CANONICAL_TOKENS)}

_count_matrix = np.zeros((_n_tok, _n_tok), dtype=float)
for (_from, _to), _cnt in _transition_counts.items():
    if _from in _tok_idx and _to in _tok_idx:
        _count_matrix[_tok_idx[_from], _tok_idx[_to]] += _cnt

_row_sums    = _count_matrix.sum(axis=1, keepdims=True)
_prob_matrix = np.where(_row_sums > 0, _count_matrix / _row_sums, 0.0)

transition_matrix_df = pd.DataFrame(
    _prob_matrix,
    index=CANONICAL_TOKENS,
    columns=CANONICAL_TOKENS
)
transition_matrix_df.index.name   = "from_token"
transition_matrix_df.columns.name = "to_token"

_tm_md = transition_matrix_df.round(4).reset_index()
print("\n  Row-normalized transition probabilities (from → to):\n")
print(_tm_md.to_string(index=False))

print(f"\n  transition_matrix_df: {transition_matrix_df.shape[0]} x {transition_matrix_df.shape[1]}")
print(f"  Tokens with outgoing transitions: {int((transition_matrix_df.sum(axis=1) > 0).sum())}/{_n_tok}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRANSITION MATRIX HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering transition matrix heatmap...")

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

_cmap = mcolors.LinearSegmentedColormap.from_list(
    "zerve_heat",
    [BG, "#1b4f72", BLUE, YELLOW],
    N=256
)
_vmax = max(float(_prob_matrix.max()), 0.01)

_im = ax.imshow(
    transition_matrix_df.values,
    cmap=_cmap, aspect="auto",
    vmin=0, vmax=_vmax
)

ax.set_xticks(range(_n_tok))
ax.set_yticks(range(_n_tok))
ax.set_xticklabels(CANONICAL_TOKENS, rotation=45, ha="right", color=FG, fontsize=9)
ax.set_yticklabels(CANONICAL_TOKENS, color=FG, fontsize=9)

for _i in range(_n_tok):
    for _j in range(_n_tok):
        _val = transition_matrix_df.values[_i, _j]
        if _val > 0.001:
            _txt_color = BG if _val > 0.45 else FG
            ax.text(_j, _i, f"{_val:.2f}", ha="center", va="center",
                    fontsize=7, color=_txt_color, fontweight="bold")

_cbar = fig.colorbar(_im, ax=ax, fraction=0.035, pad=0.02)
_cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)
_cbar.set_label("Transition Probability", color=FG, fontsize=10)

ax.set_title(
    "Markov Transition Probability Matrix\n(from-token row \u2192 to-token column, row-normalized)",
    color=FG, fontsize=13, fontweight="bold", pad=14
)
ax.set_xlabel("TO token", color=FG2, fontsize=11)
ax.set_ylabel("FROM token", color=FG2, fontsize=11)

for _i in range(_n_tok + 1):
    ax.axhline(_i - 0.5, color=BG, lw=0.8)
    ax.axvline(_i - 0.5, color=BG, lw=0.8)

ax.spines[:].set_visible(False)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  SEQUENCE ANALYSIS COMPLETE")
print("="*80)
print(f"  session_sequences_df : {len(session_sequences_df):,} rows")
print(f"  transition_matrix_df : {transition_matrix_df.shape[0]} x {transition_matrix_df.shape[1]}")
print(f"  Total sessions       : {total_sessions:,}")
print(f"  Total unique seqs    : {len(_overall_counts):,}")
print(f"  Top-1 sequence       : {_top20_overall[0][0]} (n={_top20_overall[0][1]:,})")
print("="*80)
