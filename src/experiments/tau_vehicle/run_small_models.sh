#!/usr/bin/env bash
# Small-model runs on the vehicle domain (step 1 of the vehicle+telecom plan, 2026-09-14).
# Usage: bash run_small_models.sh <MODEL_TAG> <COND> [NUM_TRIALS] [CONCURRENCY]
#   MODEL_TAG: nano     -> gpt-5.4-nano              (OpenAI, closed small anchor; OPENAI_API_KEY)
#              q35_9b   -> openrouter/qwen/qwen3.5-9b (2026-class open dense 9B; OPENROUTER_API_KEY)   <- primary
#              q3_8b_or -> openrouter/qwen/qwen3-8b   (2025-class open dense 8B; single upstream (Alibaba), 429-prone)
#              q3_14b   -> openrouter/qwen/qwen3-14b  (2025-class open dense 14B; DeepInfra/Alibaba)     <- used instead of qwen3-8b
#              min8b    -> openrouter/mistralai/ministral-8b-2512 (FAILS in tau2: Mistral returns 422 extra_forbidden)
#              q3_8b    -> dashscope/qwen3-8b        (2025-class open dense 8B; DASHSCOPE_API_KEY + DASHSCOPE_API_BASE intl)
#              q35_35a3 -> dashscope/qwen3.5-35b-a3b (2026-class open MoE, 3B active)
#              q36_35a3 -> dashscope/qwen3.6-35b-a3b (optional, 2026-04 MoE, 3B active)
#              q3_30a3  -> dashscope/qwen3-30b-a3b-instruct-2507 (optional, 2025 MoE, 3B active)
#   Open Qwen models run in non-thinking mode (OpenRouter: extra_body.reasoning.enabled=false; DashScope:
#   extra_body.enable_thinking=false): the on-device regime, and DashScope requires streaming when thinking is on.
#   COND:      A (pull, llm_agent) | B (push, llm_agent_vehicle_ctx) | Gp (pull+gate) | G (push+gate) | ALL
# Everything else mirrors the 2026-09-09 main run: user sim gpt-5.4-mini, temperature 0, seed 300, 8 trials, concurrency 6.
set -u
LOG=${LOG:-/private/tmp/claude-501/-Users-cholhwan-Documents-ai-tau2-bench/254ea8e6-244c-4bbd-92ee-4dd6c2f70815/scratchpad}
TAG=$1; COND=$2; TRIALS=${3:-8}; CONC=${4:-6}
QWEN_ARGS='{"temperature": 0.0, "extra_body": {"enable_thinking": false}}'
OR_ARGS='{"temperature": 0.0, "extra_body": {"reasoning": {"enabled": false}}}'
case "$TAG" in
  nano)     MODEL=gpt-5.4-nano; AGENT_ARGS='{"temperature": 0.0}' ;;
  q35_9b)   MODEL=openrouter/qwen/qwen3.5-9b; AGENT_ARGS=$OR_ARGS ;;
  q3_8b_or) MODEL=openrouter/qwen/qwen3-8b; AGENT_ARGS='{"temperature": 0.0, "num_retries": 5, "extra_body": {"reasoning": {"enabled": false}}}' ;;
  q3_14b)   MODEL=openrouter/qwen/qwen3-14b; AGENT_ARGS='{"temperature": 0.0, "num_retries": 3, "extra_body": {"reasoning": {"enabled": false}}}' ;;
  q3_8b)    MODEL=dashscope/qwen3-8b; AGENT_ARGS=$QWEN_ARGS ;;
  q35_35a3) MODEL=dashscope/qwen3.5-35b-a3b; AGENT_ARGS=$QWEN_ARGS ;;
  q36_35a3) MODEL=dashscope/qwen3.6-35b-a3b; AGENT_ARGS=$QWEN_ARGS ;;
  q3_30a3)  MODEL=dashscope/qwen3-30b-a3b-instruct-2507; AGENT_ARGS=$QWEN_ARGS ;;
  *) echo "unknown model tag $TAG"; exit 1 ;;
esac
agent_for() { case "$1" in A) echo llm_agent;; B) echo llm_agent_vehicle_ctx;; Gp) echo llm_agent_vehicle_gate;; G) echo llm_agent_vehicle_ctx_gate;; *) echo ""; esac; }
run_one() {
  c=$1; a=$(agent_for "$c"); [ -z "$a" ] && { echo "unknown cond $c"; return 1; }
  name="vehicle_sm_${TAG}_${c}"
  echo "[$(date +%H:%M:%S)] start $name model=$MODEL agent=$a trials=$TRIALS conc=$CONC"
  uv run tau2 run --domain vehicle --log-level WARNING --save-to "$name" \
    --agent "$a" --agent-llm "$MODEL" --agent-llm-args "$AGENT_ARGS" --user-llm gpt-5.4-mini \
    --num-trials "$TRIALS" --max-concurrency "$CONC" > "$LOG/$name.log" 2>&1
  echo "exit=$?" >> "$LOG/$name.log"
  echo "[$(date +%H:%M:%S)] done  $name exit=$(tail -1 "$LOG/$name.log")"
}
if [ "$COND" = ALL ]; then for c in A B Gp G; do run_one "$c"; done; else run_one "$COND"; fi
