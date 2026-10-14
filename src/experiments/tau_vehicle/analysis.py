"""Analysis for the vehicle-context experiment.

usage:
  python -m experiments.tau_vehicle.analysis A=data/simulations/vehicle_A/results.json B=... [--out results.md] [--json out.json]

Per condition: pass^k by task type (S/C/N), twin consistency, safety violations,
failure taxonomy, efficiency (turns, tool calls, state reads, clarifications,
tokens, cost), latency (LLM call time and user-perceived response time),
split-half noise floor, cluster-bootstrap CI for B-A, and G-theory variance
components for the S tasks.
"""

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from experiments.tau_vehicle.violations import WRITE_TOOLS, classify_failure, has_pseudo_tool_calls, replay
from tau2.metrics.agent_metrics import pass_hat_k

STATE_TOOL = "get_vehicle_state"


def meta(task: dict) -> dict:
    return dict(kv.split("=") for kv in task["description"]["notes"].replace("meta: ", "").split())


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def sim_metrics(task: dict, sim: dict) -> dict:
    msgs = sim["messages"]
    m = meta(task)
    r = (sim.get("reward_info") or {}).get("reward") or 0.0
    out = {
        "task_id": sim["task_id"], "trial": sim.get("trial", 0), "type": m["type"], "rule": m["rule"],
        "pair": m["pair"], "variant": m["variant"], "success": r >= 1.0, "termination": sim["termination_reason"],
        "duration_s": sim["duration"], "agent_cost": sim.get("agent_cost") or 0.0, "user_cost": sim.get("user_cost") or 0.0,
        "agent_msgs": 0, "agent_utterances": 0, "tool_call_msgs": 0, "tool_calls": 0, "state_reads": 0,
        "write_calls": 0, "tool_errors": 0, "user_msgs": 0, "clarifications": 0,
        "agent_prompt_tokens": 0, "agent_completion_tokens": 0, "user_prompt_tokens": 0, "user_completion_tokens": 0,
        "agent_cached_tokens": 0, "gate_revise": 0, "gate_block": 0,
        "llm_call_s": [], "response_s": [], "first_write_s": None,
    }
    first_user_ts, first_write_ts, seen_write, pending_user_ts = None, None, False, None
    for msg in msgs:
        role = msg.get("role")
        if role == "user":
            out["user_msgs"] += 1
            pending_user_ts = ts(msg["timestamp"]) if msg.get("timestamp") else None
            first_user_ts = first_user_ts or pending_user_ts
            u = msg.get("usage") or {}
            out["user_prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            out["user_completion_tokens"] += u.get("completion_tokens", 0) or 0
        elif role == "assistant":
            out["agent_msgs"] += 1
            u = msg.get("usage") or {}
            out["agent_prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            out["agent_completion_tokens"] += u.get("completion_tokens", 0) or 0
            raw = msg.get("raw_data") or {}
            out["agent_cached_tokens"] += (((raw.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
            for ev in raw.get("safety_gate") or []:
                if ev.get("action") == "REVISE":
                    out["gate_revise"] += 1
                elif ev.get("action") == "BLOCK":
                    out["gate_block"] += 1
            if msg.get("generation_time_seconds") is not None:
                out["llm_call_s"].append(msg["generation_time_seconds"])
            calls = msg.get("tool_calls") or []
            if calls:
                out["tool_call_msgs"] += 1
                out["tool_calls"] += len(calls)
                for tc in calls:
                    if tc["name"] == STATE_TOOL:
                        out["state_reads"] += 1
                    if tc["name"] in WRITE_TOOLS:
                        out["write_calls"] += 1
                        if not seen_write and first_user_ts and msg.get("timestamp"):
                            first_write_ts = ts(msg["timestamp"])
                            seen_write = True
            if msg.get("content"):
                out["agent_utterances"] += 1
                is_greeting = msg.get("usage") is None and out["user_msgs"] == 0
                if "?" in msg["content"] and not seen_write and not is_greeting:
                    out["clarifications"] += 1
                if pending_user_ts is not None and msg.get("timestamp"):
                    out["response_s"].append((ts(msg["timestamp"]) - pending_user_ts).total_seconds())
                    pending_user_ts = None
        elif role == "tool":
            if msg.get("error"):
                out["tool_errors"] += 1
    if first_write_ts and first_user_ts:
        out["first_write_s"] = (first_write_ts - first_user_ts).total_seconds()
    out["agent_msgs"] -= 1 if msgs and msgs[0].get("role") == "assistant" and not msgs[0].get("usage") else 0  # greeting
    return out


def load_condition(path: str) -> tuple[list[dict], dict]:
    res = json.loads(Path(path).read_text())
    tasks = {t["id"]: t for t in res["tasks"]}
    rows = []
    for sim in res["simulations"]:
        task = tasks[sim["task_id"]]
        row = sim_metrics(task, sim)
        rep = replay(task, sim)
        row["violations"] = [v[0] for v in rep.violations]
        row["invalid_calls"] = rep.invalid_calls
        row["pseudo_tool_calls"] = 1 if has_pseudo_tool_calls(sim) else 0
        row["failure"] = None if row["success"] else classify_failure(task, sim, rep)
        rows.append(row)
    return rows, tasks


def pass_k_table(rows: list[dict], ks=(1, 2, 4, 8)) -> dict:
    """pass^k averaged over tasks, per type and overall."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["type"], r["task_id"])].append(r["success"])
    n_trials = min(len(v) for v in by_task.values())
    out = {}
    for typ in ("ALL", "S", "C", "N"):
        vals = {k: [] for k in ks if k <= n_trials}
        for (t, _), succ in by_task.items():
            if typ != "ALL" and t != typ:
                continue
            succ = succ[:n_trials]
            for k in vals:
                vals[k].append(pass_hat_k(n_trials, sum(succ), k))
        out[typ] = {f"pass^{k}": (st.mean(v) if v else float("nan")) for k, v in vals.items()}
        out[typ]["n_tasks"] = len([1 for (t, _) in by_task if typ == "ALL" or t == typ])
    out["n_trials"] = n_trials
    return out


def twin_consistency(rows: list[dict]) -> float:
    """Fraction of (pair, trial) where both twin variants succeeded (S tasks only)."""
    by = defaultdict(dict)
    for r in rows:
        if r["type"] == "S":
            by[(r["pair"], r["trial"])][r["variant"]] = r["success"]
    both = [all(v.values()) for v in by.values() if len(v) == 2]
    return st.mean(both) if both else float("nan")


def efficiency(rows: list[dict]) -> dict:
    keys = ["agent_msgs", "agent_utterances", "tool_calls", "state_reads", "write_calls", "tool_errors",
            "user_msgs", "clarifications", "agent_prompt_tokens", "agent_completion_tokens", "agent_cost", "duration_s",
            "gate_revise", "gate_block", "pseudo_tool_calls"]
    out = {k: st.mean(r[k] for r in rows) for k in keys}
    tot_prompt = sum(r["agent_prompt_tokens"] for r in rows)
    out["cache_hit_ratio"] = (sum(r["agent_cached_tokens"] for r in rows) / tot_prompt) if tot_prompt else float("nan")
    llm = [x for r in rows for x in r["llm_call_s"]]
    resp = [x for r in rows for x in r["response_s"]]
    fw = [r["first_write_s"] for r in rows if r["first_write_s"] is not None]
    out["llm_call_p50"] = st.median(llm) if llm else float("nan")
    out["llm_call_p95"] = pct(llm, 95)
    out["response_p50"] = st.median(resp) if resp else float("nan")
    out["response_p95"] = pct(resp, 95)
    out["response_sum_per_sim"] = st.mean(sum(r["response_s"]) for r in rows)
    out["first_write_p50"] = st.median(fw) if fw else float("nan")
    out["violation_rate"] = st.mean(1.0 if r["violations"] else 0.0 for r in rows)
    out["violations_by_rule"] = dict(sorted(count_rules(rows).items()))
    return out


def count_rules(rows):
    c = defaultdict(int)
    for r in rows:
        for v in set(r["violations"]):
            c[v] += 1
    return c


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))
    return xs[i]


def failure_table(rows: list[dict]) -> dict:
    c = defaultdict(int)
    for r in rows:
        if not r["success"]:
            c[r["failure"]] += 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def split_half(rows: list[dict]) -> dict:
    """Noise floor: pass^k(first half of trials) - pass^k(second half), per type."""
    by_task = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["trial"]):
        by_task[(r["type"], r["task_id"])].append(r["success"])
    n = min(len(v) for v in by_task.values())
    h = n // 2
    if h == 0:
        return {}
    out = {}
    for typ in ("ALL", "S", "C", "N"):
        d1, d2 = [], []
        for (t, _), s in by_task.items():
            if typ != "ALL" and t != typ:
                continue
            a, b = s[:h], s[h:2 * h]
            d1.append(st.mean(a)); d2.append(st.mean(b))
        if d1:
            out[typ] = {"half1_pass1": st.mean(d1), "half2_pass1": st.mean(d2), "abs_diff": abs(st.mean(d1) - st.mean(d2))}
    return out


def bootstrap_diff(rows_a, rows_b, typ="S", n_boot=2000, seed=0) -> dict:
    """Cluster bootstrap over tasks: CI for mean success(B) - mean success(A)."""
    rnd = random.Random(seed)
    ta = defaultdict(list); tb = defaultdict(list)
    for r in rows_a:
        if typ == "ALL" or r["type"] == typ: ta[r["task_id"]].append(r["success"])
    for r in rows_b:
        if typ == "ALL" or r["type"] == typ: tb[r["task_id"]].append(r["success"])
    tasks = sorted(set(ta) & set(tb))
    if not tasks:
        return {}
    diffs = []
    for _ in range(n_boot):
        sample = [rnd.choice(tasks) for _ in tasks]
        diffs.append(st.mean(st.mean(tb[t]) for t in sample) - st.mean(st.mean(ta[t]) for t in sample))
    diffs.sort()
    point = st.mean(st.mean(tb[t]) for t in tasks) - st.mean(st.mean(ta[t]) for t in tasks)
    return {"diff": point, "ci95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]), "n_tasks": len(tasks)}


def g_theory(conds: dict[str, list[dict]], typ="S") -> dict:
    """Variance components for a balanced task x condition design with trials as replications."""
    y = defaultdict(list)
    for cname, rows in conds.items():
        for r in rows:
            if typ == "ALL" or r["type"] == typ:
                y[(r["task_id"], cname)].append(1.0 if r["success"] else 0.0)
    tasks = sorted({t for t, _ in y}); cs = sorted({c for _, c in y})
    n_r = min(len(v) for v in y.values())
    if len(cs) < 2 or n_r < 2:
        return {}
    cell = {k: v[:n_r] for k, v in y.items()}
    n_t, n_c = len(tasks), len(cs)
    gm = st.mean(x for v in cell.values() for x in v)
    mt = {t: st.mean(x for c in cs for x in cell[(t, c)]) for t in tasks}
    mc = {c: st.mean(x for t in tasks for x in cell[(t, c)]) for c in cs}
    mtc = {k: st.mean(v) for k, v in cell.items()}
    ss_t = n_c * n_r * sum((mt[t] - gm) ** 2 for t in tasks)
    ss_c = n_t * n_r * sum((mc[c] - gm) ** 2 for c in cs)
    ss_tc = n_r * sum((mtc[(t, c)] - mt[t] - mc[c] + gm) ** 2 for t in tasks for c in cs)
    ss_e = sum((x - mtc[k]) ** 2 for k, v in cell.items() for x in v)
    ms_t = ss_t / (n_t - 1); ms_c = ss_c / (n_c - 1); ms_tc = ss_tc / ((n_t - 1) * (n_c - 1)); ms_e = ss_e / (n_t * n_c * (n_r - 1))
    v_e = ms_e
    v_tc = max(0.0, (ms_tc - ms_e) / n_r)
    v_t = max(0.0, (ms_t - ms_tc) / (n_c * n_r))
    v_c = max(0.0, (ms_c - ms_tc) / (n_t * n_r))
    tot = v_t + v_c + v_tc + v_e or 1.0
    return {"task": v_t / tot, "condition": v_c / tot, "task_x_condition": v_tc / tot, "trial_residual": v_e / tot,
            "n_tasks": n_t, "n_conditions": n_c, "n_trials": n_r}


def fmt(x, nd=3):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def report(conds: dict[str, list[dict]]) -> str:
    L = []
    names = list(conds)
    L.append("## 1. Accuracy: pass^k by task type\n")
    L.append("| condition | type | n_tasks | pass^1 | pass^2 | pass^4 | pass^8 |")
    L.append("|---|---|---|---|---|---|---|")
    pk = {c: pass_k_table(rows) for c, rows in conds.items()}
    for c in names:
        for typ in ("ALL", "S", "C", "N"):
            row = pk[c][typ]
            L.append(f"| {c} | {typ} | {row['n_tasks']} | " + " | ".join(fmt(row.get(f'pass^{k}', float('nan'))) for k in (1, 2, 4, 8)) + " |")
    L.append(f"\nTrials per task: {', '.join(f'{c}={pk[c]['n_trials']}' for c in names)}\n")
    L.append("| condition | twin consistency (S pairs, both variants pass) | safety violation rate | violations by rule |")
    L.append("|---|---|---|---|")
    eff = {c: efficiency(rows) for c, rows in conds.items()}
    for c in names:
        L.append(f"| {c} | {fmt(twin_consistency(conds[c]))} | {fmt(eff[c]['violation_rate'])} | {eff[c]['violations_by_rule']} |")
    L.append("\n## 2. Failure taxonomy (failed simulations)\n")
    L.append("| condition | " + " | ".join(names and ["count by type"]) + " |")
    L.append("|---|---|")
    for c in names:
        L.append(f"| {c} | {failure_table(conds[c])} |")
    L.append("\n## 3. Efficiency and latency (means per simulation unless noted)\n")
    keys = ["agent_msgs", "agent_utterances", "tool_calls", "state_reads", "write_calls", "tool_errors", "user_msgs",
            "clarifications", "agent_prompt_tokens", "agent_completion_tokens", "cache_hit_ratio", "agent_cost", "duration_s",
            "gate_revise", "gate_block", "pseudo_tool_calls",
            "llm_call_p50", "llm_call_p95", "response_p50", "response_p95", "response_sum_per_sim", "first_write_p50"]
    L.append("| metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for k in keys:
        nd = 0 if "token" in k else (4 if "cost" in k else 2)
        L.append(f"| {k} | " + " | ".join(fmt(eff[c][k], nd) for c in names) + " |")
    L.append("\n## 4. Noise floor (split-half within condition) and B-A bootstrap CI\n")
    for c in names:
        L.append(f"- {c} split-half |pass^1 diff| by type: " + ", ".join(f"{t}={fmt(v['abs_diff'])}" for t, v in split_half(conds[c]).items()))
    if len(names) >= 2:
        a, b = names[0], names[1]
        for typ in ("ALL", "S", "C", "N"):
            bd = bootstrap_diff(conds[a], conds[b], typ)
            if bd:
                L.append(f"- {b}-{a} mean success ({typ}, {bd['n_tasks']} tasks): {fmt(bd['diff'])} 95% CI [{fmt(bd['ci95'][0])}, {fmt(bd['ci95'][1])}]")
        L.append("\n## 5. G-theory variance components (task x condition, trials as replications)\n")
        for typ in ("S", "ALL"):
            g = g_theory(conds, typ)
            if g:
                L.append(f"- {typ}: task={fmt(g['task'])}, condition={fmt(g['condition'])}, task x condition={fmt(g['task_x_condition'])}, trial residual={fmt(g['trial_residual'])} (n_tasks={g['n_tasks']}, n_trials={g['n_trials']})")
    L.append("\n## 6. Per-task pass^1\n")
    L.append("| task | type | " + " | ".join(names) + " |")
    L.append("|---|---|" + "---|" * len(names))
    per = {c: defaultdict(list) for c in names}
    types = {}
    for c in names:
        for r in conds[c]:
            per[c][r["task_id"]].append(r["success"]); types[r["task_id"]] = r["type"]
    for t in sorted(types, key=lambda x: (types[x], x)):
        L.append(f"| {t} | {types[t]} | " + " | ".join(fmt(st.mean(per[c][t])) if per[c][t] else "-" for c in names) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path/to/results.json")
    ap.add_argument("--out")
    ap.add_argument("--json")
    args = ap.parse_args()
    conds = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        conds[name], _ = load_condition(path)
    text = report(conds)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps({c: rows for c, rows in conds.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
