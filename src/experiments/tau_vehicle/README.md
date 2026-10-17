# tau_vehicle — vehicle-state context injection experiment

Design: `PLAN-vehicle-context.md` (repo root, v0.3 block). Results: `experiment-notes/RESULTS-vehicle-context.md`. Run log: `NIGHT-LOG-2026-09-09.md`.

| file | role |
|---|---|
| `make_tasks.py` | generates `data/tau2/domains/vehicle/tasks.json` + `split_tasks.json` (45 tasks: S 20 twins, C 4 conflicts, N 21 controls) |
| `validate_tasks.py` | offline gate: replays reference trajectories, no-op pass set, twin hash check |
| `context_agent.py` | condition B: `LLMAgent` + live `<vehicle_state>` block re-rendered every turn (`llm_agent_vehicle_ctx`) |
| `ablation_agents.py` | D: push without the `get_vehicle_state` tool (`llm_agent_vehicle_ctx_noread`) |
| `gate_agents.py` | B2 tail-placed block (`llm_agent_vehicle_ctx_tail`), B3 block inside the last message (`llm_agent_vehicle_ctx_lastmsg`), deterministic pre-write safety gate: Gp pull+gate (`llm_agent_vehicle_gate`), G push+gate (`llm_agent_vehicle_ctx_gate`) |
| `plot_results.py` | figures from `analysis.py --json` rows (`uv run --with matplotlib ...`) |
| `run_followups.sh` | follow-up run presets (D, gpt-5.4 agent, gpt-5.4 user, low-concurrency latency) |
| `noop_agent.py` | do-nothing agent (`noop_agent`) for the trivial-agent baseline |
| `violations.py` | replays tool calls on the initial state and applies S1-S6 predicates; failure taxonomy |
| `analysis.py` | pass^k by type, twin consistency, violations, failure taxonomy, efficiency, latency, split-half, bootstrap CI, G-theory |
| `inspect_runs.py` | compact trajectory printer |
| `tests/` | isolation tests for the context agent |

Commands:

```bash
uv run python -m experiments.tau_vehicle.make_tasks
uv run python -m experiments.tau_vehicle.validate_tasks
uv run --with pytest python -m pytest tests/test_domains/test_vehicle src/experiments/tau_vehicle/tests -q
uv run tau2 run --domain vehicle --agent llm_agent             --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 8 --max-concurrency 6 --save-to vehicle_main_A
uv run tau2 run --domain vehicle --agent llm_agent_vehicle_ctx --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 8 --max-concurrency 6 --save-to vehicle_main_B
uv run python -m experiments.tau_vehicle.analysis A=data/simulations/vehicle_main_A/results.json B=data/simulations/vehicle_main_B/results.json --out experiment-notes/RESULTS-tables.md
```
