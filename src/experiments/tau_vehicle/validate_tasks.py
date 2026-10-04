"""Offline task validation (no LLM).

For every task: apply initialization_actions to a fresh env, record the
initial DB hash, replay the reference trajectory, check that no tool call
errors, and record the target hash. Reports:
- tasks whose reference trajectory errors (must be 0)
- no-op pass set: tasks whose target hash == initial hash (a do-nothing agent passes)
- twin pairs whose two variants end in the same hash (twin design broken)
"""

from collections import defaultdict

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.vehicle.environment import get_environment, get_tasks


def meta(task) -> dict:
    notes = task.description.notes or ""
    return dict(kv.split("=") for kv in notes.replace("meta: ", "").split())


def env_after_init(task):
    env = get_environment()
    acts = task.initial_state.initialization_actions if task.initial_state else None
    if acts:
        env.run_env_function_calls([EnvFunctionCall.model_validate(a.model_dump()) for a in acts])
    return env


def main():
    tasks = get_tasks()
    errors, noop_pass, hashes = [], [], {}
    for t in tasks:
        env = env_after_init(t)
        h0 = env.get_db_hash()
        for a in t.evaluation_criteria.actions:
            resp = env.get_response(ToolCall(id=a.action_id, name=a.name, arguments=a.arguments, requestor="assistant"))
            if resp.error:
                errors.append((t.id, a.name, resp.content))
        h1 = env.get_db_hash()
        if t.evaluation_criteria.env_assertions:
            for ea in t.evaluation_criteria.env_assertions:
                ok = env.run_env_assertion(ea, raise_assertion_error=False)
                if not ok:
                    errors.append((t.id, ea.func_name, "assertion failed on reference trajectory"))
        hashes[t.id] = (h0, h1)
        if h0 == h1:
            noop_pass.append(t.id)
    pairs = defaultdict(list)
    for t in tasks:
        m = meta(t)
        if m["type"] == "S":
            pairs[m["pair"]].append(t.id)
    broken = [p for p, ids in pairs.items() if len(ids) == 2 and hashes[ids[0]][1] == hashes[ids[1]][1]]
    print(f"tasks={len(tasks)} reference_errors={len(errors)}")
    for e in errors:
        print("  ERROR", e)
    print(f"noop_pass={len(noop_pass)}/{len(tasks)} ({100*len(noop_pass)/len(tasks):.0f}%): {noop_pass}")
    print(f"twin_pairs={len(pairs)} broken_pairs={broken}")
    return len(errors) == 0 and not broken


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
