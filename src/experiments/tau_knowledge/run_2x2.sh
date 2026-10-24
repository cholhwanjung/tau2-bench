#!/usr/bin/env bash
# 2x2 runs for the banking_knowledge facts experiment (PLAN-banking-facts.md), gpt-5.4-mini only.
# Usage: bash src/experiments/tau_knowledge/run_2x2.sh <smoke|main> <ARM...|ALL>
#   smoke: 10 stratified tasks x 2 trials (regime and unit-cost check; plan says A and C).
#          The first tasks by id are all recommend/transfer tasks while 69 of 97 tasks are
#          unlock-chain tasks, so the smoke samples clusters: recommend 001 003, transfer 004,
#          give-tool 017 019, unlock-chain 026 035 (1 tool) 039 060 (3 tools) 043 (7 tools).
#   main:  97 tasks x 4 trials
#   ARM:   A prose+openai_embeddings | B facts+openai_embeddings | C prose+golden_retrieval | D facts+golden_retrieval
#          T title-prepended prose+openai_embeddings (control: the index embeds content only, and the
#            facts rewrite copies the title's product name into the content; make_title_corpus.py)
# Fixed: agent gpt-5.4-mini, user gpt-5.4-mini, temperature 0, seed 300, concurrency $CONC (default 6).
# Results: data/simulations/bk_facts_<mode>_<arm>/results.json, log next to it. Re-running resumes.
set -u
cd "$(dirname "$0")/../../.."
MODE=${1:?mode smoke|main}; shift
CONC=${CONC:-6}
TRIALS=${TRIALS:-4}   # main: TRIALS=2 for a first pass; re-run with TRIALS=4 later and only trials 3-4 are added (resume)
FACTS=${FACTS:-data/tau2/domains/banking_knowledge/documents_facts}
TITLE=${TITLE:-data/tau2/domains/banking_knowledge/documents_title}
case "$MODE" in
  smoke) EXTRA="--task-ids task_001 task_003 task_004 task_017 task_019 task_026 task_035 task_039 task_043 task_060 --num-trials 2" ;;
  main)  EXTRA="--num-trials $TRIALS" ;;
  *) echo "mode must be smoke|main"; exit 1 ;;
esac
ARMS="$*"; [ "$ARMS" = ALL ] && ARMS="A B C D"
[ -z "$ARMS" ] && { echo "give at least one arm (A B C D) or ALL"; exit 1; }
for arm in $ARMS; do
  case "$arm" in
    A) RC=openai_embeddings; KW="" ;;
    B) RC=openai_embeddings; KW="{\"documents_dir\": \"$FACTS\"}" ;;
    C) RC=golden_retrieval;  KW="" ;;
    D) RC=golden_retrieval;  KW="{\"documents_dir\": \"$FACTS\"}" ;;
    T) RC=openai_embeddings; KW="{\"documents_dir\": \"$TITLE\"}" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  if [ "$arm" = T ]; then
    [ "$(ls "$TITLE"/*.json 2>/dev/null | wc -l)" -eq 698 ] || { echo "title corpus missing: run make_title_corpus first"; exit 1; }
  elif [ -n "$KW" ]; then
    # B searches the whole corpus, so every prose document needs a validated facts twin;
    # D only inlines required documents, so the required set (make_facts --required-only) is enough.
    VAL=""; [ "$arm" = D ] && VAL="--required-only"
    uv run python -m experiments.tau_knowledge.make_facts --validate-only $VAL --out "$FACTS" > /dev/null 2>&1 \
      || { echo "facts corpus at $FACTS fails validation for arm $arm (run make_facts${VAL:+ $VAL}, then --validate-only$VAL)"; exit 1; }
  fi
  name="bk_facts_${MODE}_${arm}"
  mkdir -p data/simulations
  echo "[$(date +%H:%M:%S)] start $name retrieval=$RC kwargs=${KW:-none} $EXTRA conc=$CONC"
  uv run tau2 run --domain banking_knowledge --log-level WARNING --save-to "$name" --auto-resume \
    --agent llm_agent --agent-llm gpt-5.4-mini --agent-llm-args '{"temperature": 0.0}' \
    --user-llm gpt-5.4-mini --user-llm-args '{"temperature": 0.0}' --seed 300 \
    --retrieval-config "$RC" ${KW:+--retrieval-config-kwargs "$KW"} \
    --max-concurrency "$CONC" $EXTRA > "data/simulations/$name.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] done  $name exit=$rc"
done
