"""Control corpus for the facts experiment: prose documents with the title prepended to the content.

usage:
  uv run python -m experiments.tau_knowledge.make_title_corpus [--out DIR]

The dense index embeds ``content`` only, so a title such as "Business Silver Rewards
Card: Exceptions and Exclusions" is invisible to retrieval in the prose corpus, while
the facts rewrite copies the product name into the content. This corpus (arm T in
run_2x2.sh) separates that title effect from the schema structure: same ids, titles
and file names; content = "# <title>\\n\\n<prose content>". No LLM call.
"""

import argparse
import json
from pathlib import Path

from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DATA_DIR, KNOWLEDGE_DOCUMENTS_DIR

DEFAULT_OUT = KNOWLEDGE_DATA_DIR / "documents_title"


def title_content(doc: dict) -> str:
    return f"# {doc['title']}\n\n{doc['content']}"


def build(src: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.json")):
        doc = json.loads(path.read_text())
        payload = {"id": doc["id"], "title": doc["title"], "content": title_content(doc)}
        (out / path.name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=KNOWLEDGE_DOCUMENTS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    print(f"wrote {build(args.src, args.out)} docs to {args.out}")


if __name__ == "__main__":
    main()
