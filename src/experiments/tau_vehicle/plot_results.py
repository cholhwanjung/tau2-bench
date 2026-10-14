"""Figures from analysis JSON (per-simulation rows per condition).

usage: uv run --with matplotlib python -m experiments.tau_vehicle.plot_results rows.json out_prefix
Produces <out_prefix>_passk.png (pass^k by type, A vs B, split-half band on A) and
<out_prefix>_efficiency.png (turns / tool calls / state reads / response latency).
"""

import json
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tau2.metrics.agent_metrics import pass_hat_k


def passk_by_type(rows, ks):
    by = defaultdict(list)
    for r in rows:
        by[(r["type"], r["task_id"])].append(r["success"])
    n = min(len(v) for v in by.values())
    out = {}
    for typ in ("S", "C", "N", "ALL"):
        out[typ] = {}
        for k in ks:
            if k > n:
                continue
            vals = [pass_hat_k(n, sum(v[:n]), k) for (t, _), v in by.items() if typ == "ALL" or t == typ]
            out[typ][k] = st.mean(vals)
    return out, n


def split_half_band(rows):
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["trial"]):
        by[(r["type"], r["task_id"])].append(r["success"])
    n = min(len(v) for v in by.values()); h = n // 2
    band = {}
    if h == 0:
        return {typ: 0.0 for typ in ("S", "C", "N", "ALL")}
    for typ in ("S", "C", "N", "ALL"):
        a = [st.mean(v[:h]) for (t, _), v in by.items() if typ == "ALL" or t == typ]
        b = [st.mean(v[h:2 * h]) for (t, _), v in by.items() if typ == "ALL" or t == typ]
        band[typ] = abs(st.mean(a) - st.mean(b)) if a and h else 0.0
    return band


def main(path, prefix):
    conds = json.load(open(path))
    names = list(conds)
    ks = (1, 2, 4, 8)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    band = split_half_band(conds[names[0]])
    for ax, typ in zip(axes, ("S", "C", "N", "ALL")):
        for name, marker in zip(names, ("o", "s", "^", "v", "D", "P", "X", "*")):
            pk, n = passk_by_type(conds[name], ks)
            xs = sorted(pk[typ]); ys = [pk[typ][k] for k in xs]
            ax.plot(xs, ys, marker=marker, label=name)
        base = passk_by_type(conds[names[0]], ks)[0][typ]
        if base:
            xs = sorted(base); ys = [base[k] for k in xs]
            ax.fill_between(xs, [y - band[typ] for y in ys], [y + band[typ] for y in ys], alpha=0.15, label=f"{names[0]} split-half band" if typ == "S" else None)
        ax.set_title(f"type {typ}"); ax.set_xlabel("k"); ax.set_xticks(list(ks)); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass^k"); axes[0].legend(fontsize=8)
    fig.suptitle("Vehicle domain: pass^k by task type — A pull · B push (system prompt) · B2 push (tail) · D push w/o read tool · Gp pull+gate · G push+gate")
    fig.tight_layout(); fig.savefig(f"{prefix}_passk.png", dpi=150)

    keys = ["agent_msgs", "tool_calls", "state_reads", "clarifications", "response_p50", "agent_prompt_tokens"]
    labels = ["agent turns", "tool calls", "state reads", "clarifying Qs", "response p50 (s)", "prompt tokens (k)"]
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 3.2))
    for ax, k, lab in zip(axes, keys, labels):
        vals = []
        for name in names:
            rows = conds[name]
            if k == "response_p50":
                xs = [x for r in rows for x in r["response_s"]]
                vals.append(st.median(xs) if xs else 0)
            elif k == "agent_prompt_tokens":
                vals.append(st.mean(r[k] for r in rows) / 1000)
            else:
                vals.append(st.mean(r[k] for r in rows))
        ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"][: len(names)])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Efficiency and latency per simulation (means; response p50 over turns)")
    fig.tight_layout(); fig.savefig(f"{prefix}_efficiency.png", dpi=150)
    print("wrote", f"{prefix}_passk.png", f"{prefix}_efficiency.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
