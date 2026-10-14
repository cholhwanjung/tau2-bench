"""Print compact trajectories from a tau2 results file.

usage: python -m experiments.tau_vehicle.inspect_runs <results.json> [--failed] [--task ID ...]
"""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def compact(sim, max_chars=160):
    lines = []
    for m in sim["messages"]:
        role = m.get("role")
        if role == "assistant":
            if m.get("tool_calls"):
                calls = "; ".join(f"{tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})" for tc in m["tool_calls"])
                lines.append(f"  A>tool {calls}")
            if m.get("content"):
                lines.append(f"  A: {m['content'][:max_chars]}")
        elif role == "user":
            lines.append(f"  U: {(m.get('content') or '')[:max_chars]}")
        elif role == "tool":
            c = (m.get("content") or "").replace("\n", " ")
            lines.append(f"  T{'!' if m.get('error') else ''}: {c[:max_chars]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--task", nargs="*")
    ap.add_argument("--max-chars", type=int, default=160)
    args = ap.parse_args()
    res = load(args.path)
    for sim in res["simulations"]:
        r = (sim.get("reward_info") or {}).get("reward")
        if args.failed and r == 1.0:
            continue
        if args.task and sim["task_id"] not in args.task:
            continue
        print(f"=== {sim['task_id']} trial={sim.get('trial')} reward={r} term={sim['termination_reason']} dur={sim['duration']:.0f}s cost={sim.get('agent_cost')}")
        print(compact(sim, args.max_chars))


if __name__ == "__main__":
    main()
