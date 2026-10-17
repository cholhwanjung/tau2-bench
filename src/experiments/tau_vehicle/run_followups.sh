#!/usr/bin/env bash
# Follow-up runs after the main A/B. Usage: bash run_followups.sh <stage>
# stages: D (push without pull), M54 (agent gpt-5.4, A and B), USER54 (user sim gpt-5.4, A and B), LAT (low-concurrency latency runs)
set -u
LOG=/private/tmp/claude-501/-Users-cholhwan-Documents-ai-tau2-bench/2627b5ee-85d9-4df3-94be-671f767d01a4/scratchpad
run() { name=$1; shift; uv run tau2 run --domain vehicle --log-level WARNING --save-to "$name" "$@" > "$LOG/$name.log" 2>&1; echo "exit=$?" >> "$LOG/$name.log"; }
case "$1" in
  D)      run vehicle_abl_D --agent llm_agent_vehicle_ctx_noread --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 8 --max-concurrency 6 ;;
  M54A)   run vehicle_m54_A --agent llm_agent --agent-llm gpt-5.4 --user-llm gpt-5.4-mini --num-trials 4 --max-concurrency 6 ;;
  M54B)   run vehicle_m54_B --agent llm_agent_vehicle_ctx --agent-llm gpt-5.4 --user-llm gpt-5.4-mini --num-trials 4 --max-concurrency 6 ;;
  U54A)   run vehicle_u54_A --agent llm_agent --agent-llm gpt-5.4-mini --user-llm gpt-5.4 --num-trials 4 --max-concurrency 6 ;;
  U54B)   run vehicle_u54_B --agent llm_agent_vehicle_ctx --agent-llm gpt-5.4-mini --user-llm gpt-5.4 --num-trials 4 --max-concurrency 6 ;;
  LATA)   run vehicle_lat_A --agent llm_agent --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 2 --max-concurrency 2 ;;
  LATB)   run vehicle_lat_B --agent llm_agent_vehicle_ctx --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 2 --max-concurrency 2 ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
