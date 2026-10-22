"""Build the facts corpus for banking_knowledge: one prose document -> one
schema-record document with the same id, title and file name (see schema.md).

usage:
  uv run python -m experiments.tau_knowledge.make_facts [--limit 5] [--ids ID ...]
  uv run python -m experiments.tau_knowledge.make_facts --validate-only
  uv run python -m experiments.tau_knowledge.make_facts --show doc_id

Each LLM response is cached on disk under data/.facts_cache keyed by
(model, schema, document, feedback), so re-runs cost nothing. A document passes
validation when every tool token (name_1234) and every numeric token of the
prose appears in the facts text; a failing document is re-extracted with the
missing tokens listed, up to --max-retries times. <out>_MANIFEST.json (beside
the directory) records the model, schema hash, per-document attempts, cost,
validation result and the share of text left in ``Other`` records. Exit status
1 if any document fails. --validate-only re-checks an existing facts directory
(every prose document present, same title and file name, tokens preserved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tau2.domains.banking_knowledge.utils import (
    KNOWLEDGE_DATA_DIR,
    KNOWLEDGE_DOCUMENTS_DIR,
    KNOWLEDGE_TASK_SET_PATH,
)
from tau2.utils.utils import DATA_DIR

SCHEMA_PATH = Path(__file__).with_name("schema.md")
DEFAULT_OUT = KNOWLEDGE_DATA_DIR / "documents_facts"
CACHE_DIR = DATA_DIR / ".facts_cache"
DEFAULT_MODEL = "gpt-5.4"

TOOL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d{4}\b")
NUM_RE = re.compile(r"\d+(?:[.,:/]\d+)*%?")
# List enumerators are not facts: "1. " / "1) " / "(1) " / "- " at line start, and inline "1) ".
LIST_MARKER_RE = re.compile(r"^\s*(?:\(?\d+[.)]|[-*])\s+|(?<=\s)\(?\d{1,2}\)\s", re.MULTILINE)
OTHER_HEADER = "### Other"

SYSTEM_TEMPLATE = (
    "You rewrite one Rho-Bank knowledge-base document into typed fact records. "
    "Follow the schema below exactly. Output only the markdown records.\n\n{schema}"
)
USER_TEMPLATE = "Document id: {id}\nTitle: {title}\n\n<prose>\n{content}\n</prose>"
FEEDBACK_TEMPLATE = (
    "\n\nYour previous rewrite dropped these tokens, which must appear verbatim: "
    "{missing}. Rewrite the whole document again and include every one of them."
)


def numeric_tokens(text: str) -> set[str]:
    """Numbers in the text, ignoring list markers and the digits of tool names."""
    return set(NUM_RE.findall(TOOL_RE.sub("", LIST_MARKER_RE.sub("", text))))


def manifest_path(out: Path) -> Path:
    """Beside the corpus, never inside it: KnowledgeBase.load reads every *.json in the dir."""
    return out.parent / f"{out.name}_MANIFEST.json"


def validate_facts(prose: str, facts: str) -> dict:
    """Every tool token and numeric token of the prose must appear in the facts."""
    missing_tools = sorted(set(TOOL_RE.findall(prose)) - set(TOOL_RE.findall(facts)))
    missing_numbers = sorted(numeric_tokens(prose) - numeric_tokens(facts))
    return {
        "ok": not missing_tools and not missing_numbers,
        "missing_tools": missing_tools,
        "missing_numbers": missing_numbers,
    }


def other_share(facts: str) -> float:
    """Fraction of facts characters that sit under '### Other' records."""
    blocks = re.split(r"(?m)^(?=### )", facts)
    other = sum(len(b) for b in blocks if b.startswith(OTHER_HEADER))
    return other / max(len(facts), 1)


def load_docs(src: Path) -> list[dict]:
    docs = []
    for path in sorted(src.glob("*.json")):
        doc = json.loads(path.read_text())
        doc["_file"] = path.name
        docs.append(doc)
    return docs


def required_doc_ids(tasks_dir: Path = KNOWLEDGE_TASK_SET_PATH) -> set[str]:
    """Union of ``required_documents`` over all tasks: the documents the golden arms inline."""
    ids: set[str] = set()
    for path in sorted(tasks_dir.glob("task_*.json")):
        ids.update(json.loads(path.read_text()).get("required_documents") or [])
    return ids


def _cache_key(model: str, schema: str, doc: dict, feedback: str) -> str:
    h = hashlib.sha256()
    for part in (model, schema, doc["id"], doc["content"], feedback):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def call_llm(model: str, system: str, user: str, key: str) -> dict:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    from litellm import completion

    from tau2.utils.llm_utils import get_response_cost, get_response_usage

    resp = completion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        num_retries=3,
    )
    rec = {
        "content": resp.choices[0].message.content or "",
        "usage": get_response_usage(resp),
        "cost": get_response_cost(resp),
        "model": model,
        "created_at": datetime.now().isoformat(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def extract_doc(doc: dict, model: str, schema: str, max_retries: int) -> dict:
    system = SYSTEM_TEMPLATE.format(schema=schema)
    user = USER_TEMPLATE.format(id=doc["id"], title=doc["title"], content=doc["content"])
    feedback, attempts, cost, prompt_tokens, completion_tokens = "", 0, 0.0, 0, 0
    while True:
        attempts += 1
        rec = call_llm(model, system, user + feedback, _cache_key(model, schema, doc, feedback))
        cost += rec.get("cost") or 0.0
        usage = rec.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0
        facts = rec["content"].strip()
        check = validate_facts(doc["content"], facts)
        if check["ok"] or attempts > max_retries:
            break
        feedback = FEEDBACK_TEMPLATE.format(missing=", ".join(check["missing_tools"] + check["missing_numbers"]))
    return {
        "id": doc["id"], "title": doc["title"], "content": facts, "file": doc["_file"],
        "attempts": attempts, "cost": cost, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "len_prose": len(doc["content"]), "len_facts": len(facts), "other_share": other_share(facts), **check,
    }


def write_doc(out: Path, rec: dict) -> None:
    payload = {"id": rec["id"], "title": rec["title"], "content": rec["content"]}
    (out / rec["file"]).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> int:
    schema = SCHEMA_PATH.read_text()
    docs = load_docs(args.src)
    if args.required_only:
        keep = required_doc_ids()
        docs = [d for d in docs if d["id"] in keep]
    if args.ids:
        docs = [d for d in docs if d["id"] in set(args.ids)]
    if args.limit:
        docs = docs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"extracting {len(docs)} docs with {args.model} -> {args.out} (workers={args.workers})")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_doc, d, args.model, schema, args.max_retries): d["id"] for d in docs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results[rec["id"]] = rec
            write_doc(args.out, rec)
            if not rec["ok"] or i % 50 == 0 or i == len(docs):
                status = "ok" if rec["ok"] else f"FAIL tools={rec['missing_tools']} numbers={rec['missing_numbers']}"
                print(f"[{i}/{len(docs)}] {rec['id']} attempts={rec['attempts']} ${rec['cost']:.4f} {status}")
    failed = sorted(r["id"] for r in results.values() if not r["ok"])
    manifest = {
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "source_dir": str(args.src),
        "scope": "required_only" if args.required_only else "all",
        "n_docs": len(results),
        "n_failed": len(failed),
        "failed": failed,
        "total_cost_usd": sum(r["cost"] for r in results.values()),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results.values()),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results.values()),
        "docs": {k: {kk: v for kk, v in r.items() if kk not in ("content", "title", "file")} for k, r in sorted(results.items())},
    }
    manifest_path(args.out).write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    shares = sorted(r["other_share"] for r in results.values())
    print(
        f"done: {len(results)} docs, {len(failed)} failed, ${manifest['total_cost_usd']:.2f}, "
        f"other_share median={shares[len(shares) // 2]:.2f} max={shares[-1]:.2f}, "
        f"len ratio facts/prose={sum(r['len_facts'] for r in results.values()) / max(1, sum(r['len_prose'] for r in results.values())):.2f}"
    )
    return 1 if failed else 0


def validate_dir(args: argparse.Namespace) -> int:
    prose = {d["id"]: d for d in load_docs(args.src)}
    facts = {d["id"]: d for d in load_docs(args.out)}
    scope = required_doc_ids() if args.required_only else set(prose)
    problems = []
    for doc_id, p in prose.items():
        if doc_id not in scope:
            continue
        f = facts.get(doc_id)
        if f is None:
            problems.append((doc_id, "missing facts document"))
            continue
        if f["title"] != p["title"] or f["_file"] != p["_file"]:
            problems.append((doc_id, "title or file name differs"))
        check = validate_facts(p["content"], f["content"])
        if not check["ok"]:
            problems.append((doc_id, f"missing tools={check['missing_tools']} numbers={check['missing_numbers']}"))
    extra = sorted(set(facts) - set(prose))
    for doc_id in extra:
        problems.append((doc_id, "not in prose corpus"))
    for doc_id, why in problems:
        print(f"FAIL {doc_id}: {why}")
    shares = sorted(other_share(f["content"]) for f in facts.values()) or [0.0]
    print(f"{len(facts)} facts docs / {len(scope)} in scope of {len(prose)} prose docs, {len(problems)} problems, "
          f"other_share median={shares[len(shares) // 2]:.2f} p90={shares[int(0.9 * (len(shares) - 1))]:.2f}")
    return 1 if problems else 0


def show(args: argparse.Namespace) -> int:
    for label, src in (("PROSE", args.src), ("FACTS", args.out)):
        matches = [d for d in load_docs(src) if d["id"] == args.show]
        print(f"===== {label} {args.show}" + (f" | {matches[0]['title']}" if matches else " (not found)"))
        if matches:
            print(matches[0]["content"])
        print()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=KNOWLEDGE_DOCUMENTS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N docs (cost/format check)")
    ap.add_argument("--ids", nargs="*", help="only these document ids")
    ap.add_argument("--required-only", action="store_true",
                    help="only the documents some task requires (enough for the golden arms C/D; arm B needs the whole corpus)")
    ap.add_argument("--validate-only", action="store_true", help="validate an existing facts dir against the prose")
    ap.add_argument("--show", metavar="DOC_ID", help="print prose and facts for one document")
    args = ap.parse_args()
    if args.show:
        sys.exit(show(args))
    if args.validate_only:
        sys.exit(validate_dir(args))
    sys.exit(build(args))


if __name__ == "__main__":
    main()
