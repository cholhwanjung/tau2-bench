"""Analysis for the banking_knowledge facts 2x2 (PLAN-banking-facts.md).

usage:
  uv run python -m experiments.tau_knowledge.analysis A=data/simulations/bk_facts_main_A/results.json B=... C=... D=... [--out experiment-notes/RESULTS-banking-facts.md] [--json rows.json]

Arms: A prose+dense, B facts+dense, C prose+golden, D facts+golden. Any names work;
the 2x2 table and the planned contrasts (B-A, D-C, C-A, D-B) appear when those
names are present. Per arm: pass^k for k <= trials, flip rate (tasks with mixed
outcomes across trials: the noise floor), document recall (required ids surfaced
by KB_search), search calls, discoverable-tool calls, agent turns, tool calls,
tokens, cache hit ratio, cost, wall-clock, termination reasons. Contrasts use a
paired cluster bootstrap over tasks. Infrastructure-error simulations are
excluded and counted.
"""

import argparse
import json
import math
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from tau2.data_model.simulation import TerminationReason
from tau2.metrics.agent_metrics import pass_hat_k

SEARCH_TOOLS = {"KB_search", "KB_search_bm25", "KB_search_dense", "grep"}
DISCOVERABLE_TOOLS = {"unlock_discoverable_agent_tool", "call_discoverable_agent_tool", "give_discoverable_user_tool"}
DOC_ID_RE = re.compile(r"^\s*ID: (\S+)", re.MULTILINE)
INFRA = TerminationReason.INFRASTRUCTURE_ERROR.value
CONTRASTS = [("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")]


def sim_metrics(task: dict, sim: dict) -> dict:
    msgs = sim.get("messages") or []
    reward = (sim.get("reward_info") or {}).get("reward")
    out = {
        "task_id": sim["task_id"], "trial": sim.get("trial", 0), "success": reward is not None and reward >= 1.0,
        "reward": reward, "termination": sim.get("termination_reason"), "duration_s": sim.get("duration") or 0.0,
        "agent_cost": sim.get("agent_cost") or 0.0, "user_cost": sim.get("user_cost") or 0.0,
        "agent_msgs": 0, "user_msgs": 0, "tool_calls": 0, "search_calls": 0, "discoverable_calls": 0, "tool_errors": 0,
        "agent_prompt_tokens": 0, "agent_completion_tokens": 0, "agent_cached_tokens": 0,
        "user_prompt_tokens": 0, "user_completion_tokens": 0, "n_required": 0, "recall": None,
    }
    search_ids = set()
    for m in msgs:
        role, u = m.get("role"), m.get("usage") or {}
        if role == "assistant":
            out["agent_msgs"] += 1 if u else 0  # the scripted greeting has no usage
            out["agent_prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            out["agent_completion_tokens"] += u.get("completion_tokens", 0) or 0
            raw_usage = (m.get("raw_data") or {}).get("usage") or {}
            out["agent_cached_tokens"] += (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            for tc in m.get("tool_calls") or []:
                out["tool_calls"] += 1
                if tc["name"] in SEARCH_TOOLS:
                    out["search_calls"] += 1
                    search_ids.add(tc.get("id"))
                if tc["name"] in DISCOVERABLE_TOOLS:
                    out["discoverable_calls"] += 1
        elif role == "user":
            out["user_msgs"] += 1
            out["user_prompt_tokens"] += u.get("prompt_tokens", 0) or 0
            out["user_completion_tokens"] += u.get("completion_tokens", 0) or 0
        elif role == "tool" and m.get("error"):
            out["tool_errors"] += 1
    required = task.get("required_documents") or []
    out["n_required"] = len(required)
    if required and search_ids:
        surfaced = set()
        for m in msgs:
            if m.get("role") == "tool" and m.get("id") in search_ids and not m.get("error"):
                surfaced.update(DOC_ID_RE.findall(m.get("content") or ""))
        out["recall"] = len(set(required) & surfaced) / len(required)
    return out


def load_condition(path: str) -> tuple[list[dict], int, dict]:
    res = json.loads(Path(path).read_text())
    tasks = {t["id"]: t for t in res["tasks"]}
    rows, infra = [], 0
    for sim in res["simulations"]:
        if sim.get("termination_reason") == INFRA or not sim.get("messages"):
            infra += 1
            continue
        rows.append(sim_metrics(tasks[sim["task_id"]], sim))
    if any(r["search_calls"] for r in rows):  # a search arm: not searching counts as recall 0
        for r in rows:
            if r["recall"] is None and r["n_required"]:
                r["recall"] = 0.0
    return rows, infra, res.get("info") or {}


def by_task(rows: list[dict], key: str = "success") -> dict[str, list]:
    d = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["trial"]):
        d[r["task_id"]].append(r[key])
    return d


def pass_k(rows: list[dict]) -> dict:
    bt = by_task(rows)
    n = min(len(v) for v in bt.values())
    out = {f"pass^{k}": st.mean(pass_hat_k(n, sum(v[:n]), k) for v in bt.values()) for k in (1, 2, 4, 8) if k <= n}
    out["n_tasks"], out["n_trials"] = len(bt), n
    out["flip_rate"] = st.mean(0 < sum(v[:n]) < n for v in bt.values()) if n > 1 else float("nan")
    out["tasks_all_fail"] = sum(1 for v in bt.values() if sum(v[:n]) == 0)
    return out


def efficiency(rows: list[dict]) -> dict:
    keys = ("agent_msgs", "user_msgs", "tool_calls", "search_calls", "discoverable_calls", "tool_errors",
            "agent_prompt_tokens", "agent_completion_tokens", "agent_cost", "user_cost", "duration_s")
    e = {k: st.mean(r[k] for r in rows) for k in keys}
    tot = sum(r["agent_prompt_tokens"] for r in rows)
    e["cache_hit_ratio"] = sum(r["agent_cached_tokens"] for r in rows) / tot if tot else float("nan")
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    e["recall"] = st.mean(recalls) if recalls else None
    e["duration_p50"] = st.median(r["duration_s"] for r in rows)
    e["total_cost"] = sum(r["agent_cost"] + r["user_cost"] for r in rows)
    return e


def bootstrap_diff(rows_a, rows_b, key="success", n_boot=2000, seed=0):
    """Paired cluster bootstrap over tasks: CI for mean(key)(B) - mean(key)(A)."""
    ta, tb = by_task(rows_a, key), by_task(rows_b, key)
    tasks = sorted(t for t in set(ta) & set(tb) if None not in ta[t] and None not in tb[t])
    if not tasks:
        return None
    d = {t: st.mean(tb[t]) - st.mean(ta[t]) for t in tasks}
    rnd = random.Random(seed)
    diffs = sorted(st.mean(d[rnd.choice(tasks)] for _ in tasks) for _ in range(n_boot))
    return {"diff": st.mean(d.values()), "ci95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]), "n_tasks": len(tasks)}


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def report(conds: dict[str, list[dict]], infra: dict[str, int], infos: dict[str, dict]) -> str:
    names = list(conds)
    pk = {c: pass_k(conds[c]) for c in names}
    eff = {c: efficiency(conds[c]) for c in names}
    kmax = min(pk[c]["n_trials"] for c in names)
    L = ["## 0. Runs\n"]
    for c in names:
        env = infos[c].get("environment_info") or {}
        cfg = {k: v for k, v in env.items() if k != "policy"}
        L.append(f"- {c}: agent={(infos[c].get('agent_info') or {}).get('llm')}, user={(infos[c].get('user_info') or {}).get('llm')}, "
                 f"num_trials={infos[c].get('num_trials')}, env={json.dumps(cfg, default=str)[:300]}")
    if {"A", "B"} <= set(names):
        pa, pb = ((infos[c].get("environment_info") or {}).get("policy") for c in ("A", "B"))
        L.append(f"- Info policy A == B (search arms share the prompt byte for byte): {pa == pb}")
    if {"A", "B", "C", "D"} <= set(names):
        def cell(c):
            s = f"{c}: {fmt(pk[c]['pass^1'])}"
            return s + (f" (pass^{kmax} {fmt(pk[c].get(f'pass^{kmax}'))})" if kmax > 1 else "")
        L += ["\n## 1. 2x2: pass^1 (pass^k at k = trials)\n", "|  | dense (`openai_embeddings`) | golden (required docs inlined) |", "|---|---|---|",
              f"| prose | {cell('A')} | {cell('C')} |", f"| facts | {cell('B')} | {cell('D')} |"]
    L += ["\n## 2. Per-arm metrics (means per simulation unless noted)\n", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    metric_rows = [
        ("n_tasks", lambda c: pk[c]["n_tasks"], 0), ("n_trials", lambda c: pk[c]["n_trials"], 0),
        ("infra_errors_excluded", lambda c: infra[c], 0), ("pass^1", lambda c: pk[c]["pass^1"], 3),
        (f"pass^{kmax}", lambda c: pk[c].get(f"pass^{kmax}"), 3),
        ("flip_rate (tasks mixed across trials)", lambda c: pk[c]["flip_rate"], 3), ("tasks_all_fail", lambda c: pk[c]["tasks_all_fail"], 0),
        ("doc_recall (search arms)", lambda c: eff[c]["recall"], 3), ("search_calls", lambda c: eff[c]["search_calls"], 2),
        ("discoverable_calls (unlock/call/give)", lambda c: eff[c]["discoverable_calls"], 2), ("agent_msgs", lambda c: eff[c]["agent_msgs"], 2),
        ("user_msgs", lambda c: eff[c]["user_msgs"], 2), ("tool_calls", lambda c: eff[c]["tool_calls"], 2), ("tool_errors", lambda c: eff[c]["tool_errors"], 2),
        ("agent_prompt_tokens", lambda c: eff[c]["agent_prompt_tokens"], 0), ("agent_completion_tokens", lambda c: eff[c]["agent_completion_tokens"], 0),
        ("cache_hit_ratio", lambda c: eff[c]["cache_hit_ratio"], 3), ("agent_cost", lambda c: eff[c]["agent_cost"], 4), ("user_cost", lambda c: eff[c]["user_cost"], 4),
        ("total_cost (sum over sims)", lambda c: eff[c]["total_cost"], 2), ("duration_s", lambda c: eff[c]["duration_s"], 1), ("duration_p50", lambda c: eff[c]["duration_p50"], 1),
    ]
    for label, get, nd in metric_rows:
        L.append(f"| {label} | " + " | ".join(fmt(get(c), nd) for c in names) + " |")
    L += ["\n## 3. Contrasts (paired cluster bootstrap over tasks, 95% CI; CI containing 0 = no difference)\n",
          "| contrast | Δ success | 95% CI | Δ recall | 95% CI | n_tasks |", "|---|---|---|---|---|---|"]
    for b, a in CONTRASTS:
        if b in conds and a in conds:
            s = bootstrap_diff(conds[a], conds[b])
            r = bootstrap_diff(conds[a], conds[b], key="recall")
            L.append(f"| {b}-{a} | {fmt(s['diff'])} | [{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] | "
                     + (f"{fmt(r['diff'])} | [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]" if r else "n/a | n/a") + f" | {s['n_tasks']} |")
    L += ["\n## 4. Termination reasons\n", "| arm | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    terms = {c: Counter(r["termination"] for r in conds[c]) for c in names}
    for t in sorted({t for c in names for t in terms[c]}):
        L.append(f"| {t} | " + " | ".join(str(terms[c].get(t, 0)) for c in names) + " |")
    L += ["\n## 5. Per-task pass^1\n", "| task | n_required | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    per = {c: by_task(conds[c]) for c in names}
    n_req = {r["task_id"]: r["n_required"] for c in names for r in conds[c]}
    for t in sorted(n_req):
        L.append(f"| {t} | {n_req[t]} | " + " | ".join(fmt(st.mean(per[c][t])) if per[c].get(t) else "-" for c in names) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path/to/results.json")
    ap.add_argument("--out")
    ap.add_argument("--json")
    args = ap.parse_args()
    conds, infra, infos = {}, {}, {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        conds[name], infra[name], infos[name] = load_condition(path)
    text = report(conds, infra, infos)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(conds, indent=1, default=str))


if __name__ == "__main__":
    main()
