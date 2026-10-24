# tau_knowledge — banking_knowledge document representation 2x2 (prose vs facts)

Design: `PLAN-banking-facts.md` (repo root). Background and failure re-analysis of the earlier memory experiment: `experiment-notes/RESEARCH-banking-knowledge-2026-11-07.md`, `experiment-notes/RETRO-tau-memory.md`.

One independent variable: the content of the 698 knowledge-base documents. `facts` is every prose document rewritten into typed records (`schema.md`) with the same id, title and file name. The agent, prompt, tools and pipeline are identical across arms; only the directory the knowledge base is loaded from changes (`--retrieval-config-kwargs '{"documents_dir": ...}'`).

| arm | corpus | retrieval config |
|---|---|---|
| A | prose (`data/tau2/domains/banking_knowledge/documents`) | `openai_embeddings` |
| B | facts (`.../documents_facts`) | `openai_embeddings` |
| C | prose | `golden_retrieval` |
| D | facts | `golden_retrieval` |
| T | title-prepended prose (`.../documents_title`, control) | `openai_embeddings` |

Arm T exists because the dense index embeds `content` only: 81% of the "X: ..." titled documents never name X in their content, and the facts rewrite copies the product name into the content. T separates that title effect from the schema structure at zero LLM cost.

| file | role |
|---|---|
| `schema.md` | the facts schema (Product / Procedure / Tool / Other), frozen before task files were read |
| `make_facts.py` | prose -> facts corpus with one cached LLM call per document; validation that every tool token and number survives; `documents_facts_MANIFEST.json` |
| `make_title_corpus.py` | control corpus: prose with `# <title>` prepended to the content (arm T), no LLM call |
| `run_2x2.sh` | `smoke` (10 stratified tasks x 2 trials) and `main` (97 x 4) runs per arm A B C D T, gpt-5.4-mini agent and user, seed 300, temperature 0 |
| `analysis.py` | 2x2 table, pass^k, flip rate, document recall, search and discoverable-tool calls, tokens, cost, wall-clock, paired bootstrap CIs for B-A, D-C, C-A, D-B |
| `tests/` | validator tests; the corpus-swap isolation tests live in `tests/test_domains/test_banking_knowledge/test_documents_dir.py` |

Core change behind it (tau2): `RetrievalVariant.documents_dir` + `resolve_variant(documents_dir=...)`, `get_knowledge_base(documents_dir)` (fails loudly on a missing or empty directory), `warm_kb_cache(documents_dir=...)` in the batch runner, and `get_or_create_docs` validating the in-process document cache against the knowledge base it is asked for (previously a warmed cache could serve another corpus to the search index).

Commands (repo root):

```bash
uv run --with pytest python -m pytest tests/test_domains/test_banking_knowledge/test_documents_dir.py src/experiments/tau_knowledge/tests -q
uv run python -m experiments.tau_knowledge.make_facts --limit 5          # format and unit-cost check (~1 cent)
uv run python -m experiments.tau_knowledge.make_facts                    # all 698 docs, gpt-5.4, cached (~$6)
uv run python -m experiments.tau_knowledge.make_facts --validate-only    # re-check an existing facts dir
uv run python -m experiments.tau_knowledge.make_facts --show doc_bank_accounts_bank_accounts_\(general\)_001
uv run python -m experiments.tau_knowledge.make_title_corpus              # arm T corpus, instant
bash src/experiments/tau_knowledge/run_2x2.sh smoke A C                  # 40 sims on 10 stratified tasks, regime check (plan: A pass^1 >= 20%)
bash src/experiments/tau_knowledge/run_2x2.sh main ALL                   # 1,552 sims (A B C D); add T for the title control (+388)
uv run python -m experiments.tau_knowledge.analysis A=data/simulations/bk_facts_main_A/results.json B=data/simulations/bk_facts_main_B/results.json C=data/simulations/bk_facts_main_C/results.json D=data/simulations/bk_facts_main_D/results.json --out experiment-notes/RESULTS-banking-facts.md
```

Minimal path (golden pair only, see `PLAN-banking-facts.md` "최소 경로"; about $17 for two trials, $31 for four):

```bash
uv run python -m experiments.tau_knowledge.make_facts --required-only      # 218 docs the tasks require (~$3)
TRIALS=2 bash src/experiments/tau_knowledge/run_2x2.sh main C              # prose golden, 194 sims: base rate + unit cost
TRIALS=2 bash src/experiments/tau_knowledge/run_2x2.sh main D              # facts golden, if C >= 20%
uv run python -m experiments.tau_knowledge.analysis C=data/simulations/bk_facts_main_C/results.json D=data/simulations/bk_facts_main_D/results.json --out experiment-notes/RESULTS-banking-facts.md
TRIALS=4 bash src/experiments/tau_knowledge/run_2x2.sh main C D            # adds trials 3-4 only (resume by task, trial, seed)
```

Notes
- `make_facts.py` exits 1 and lists the documents whose facts still miss a tool name or number after `--max-retries` (default 2). Inspect them with `--show`; `run_2x2.sh` runs `--validate-only` and refuses to start B/D while any document is missing or failing. The manifest is written beside the directory (`documents_facts_MANIFEST.json`), never inside it, because the knowledge base loads every `*.json` in the directory as a document.
- The facts corpus gets its own embedding cache entry (content hash), computed once by the runner's warm-up before the simulations start.
- `data/.facts_cache/` holds the raw LLM responses (gitignored); delete it to force re-extraction.
