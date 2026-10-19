"""documents_dir override: swapping the corpus changes document content only.

Offline: uses the first three real documents, rewritten into a temp directory
with a marker string, and the bm25 variant for the end-to-end tool check.
"""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge import retrieval as r
from tau2.domains.banking_knowledge.environment import get_environment, get_knowledge_base
from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR, KNOWLEDGE_TASK_SET_PATH
from tau2.knowledge.embeddings_cache import clear_cached_docs, set_cached_docs

MARKER = "FACTS_MARKER"


@pytest.fixture
def facts_dir(tmp_path):
    out = tmp_path / "facts"
    out.mkdir()
    ids = []
    for path in sorted(KNOWLEDGE_DOCUMENTS_DIR.glob("*.json"))[:3]:
        doc = json.loads(path.read_text())
        doc["content"] = f"### Other\n- {MARKER} {doc['id']}"
        (out / path.name).write_text(json.dumps(doc))
        ids.append(doc["id"])
    return out, ids


@pytest.fixture(autouse=True)
def _clear_docs_cache():
    clear_cached_docs()
    yield
    clear_cached_docs()


def test_resolve_variant_sets_documents_dir_without_touching_registry(facts_dir):
    out, _ = facts_dir
    variant = r.resolve_variant("openai_embeddings", documents_dir=str(out))
    assert variant.documents_dir == out
    assert r.resolve_variant("openai_embeddings").documents_dir is None
    assert r.RETRIEVAL_VARIANTS["openai_embeddings"].documents_dir is None


def test_knowledge_base_from_documents_dir_keeps_ids_and_titles(facts_dir):
    out, ids = facts_dir
    prose = get_knowledge_base()
    facts = get_knowledge_base(out)
    assert sorted(facts.documents) == sorted(ids)
    for doc_id in ids:
        assert facts.documents[doc_id].title == prose.documents[doc_id].title
        assert MARKER in facts.documents[doc_id].content
        assert facts.documents[doc_id].content != prose.documents[doc_id].content


def test_missing_or_empty_documents_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_knowledge_base(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        get_knowledge_base(tmp_path / "empty")


def test_search_variant_policy_is_byte_identical_across_corpora(facts_dir):
    out, _ = facts_dir
    variant = r.resolve_variant("openai_embeddings")
    assert r.build_policy(variant, get_knowledge_base()) == r.build_policy(variant, get_knowledge_base(out))


def test_golden_policy_differs_only_inside_required_documents(facts_dir):
    out, ids = facts_dir
    task_data = json.loads((KNOWLEDGE_TASK_SET_PATH / "task_001.json").read_text())
    task_data["required_documents"] = [ids[0]]
    task = Task.model_validate(task_data)
    variant = r.resolve_variant("golden_retrieval")
    prose_policy = r.build_policy(variant, get_knowledge_base(), task)
    facts_policy = r.build_policy(variant, get_knowledge_base(out), task)
    tag = "<required_documents>"
    assert prose_policy.split(tag)[0] == facts_policy.split(tag)[0]
    assert MARKER in facts_policy and MARKER not in prose_policy
    assert get_knowledge_base().documents[ids[0]].content in prose_policy


def test_get_or_create_docs_never_serves_another_corpus(facts_dir):
    out, ids = facts_dir
    set_cached_docs([{"id": "stale", "text": "stale", "title": "stale"}])
    docs = r.get_or_create_docs(get_knowledge_base(out))
    assert sorted(d["id"] for d in docs) == sorted(ids)
    assert all(MARKER in d["text"] for d in docs)
    assert r.get_or_create_docs(get_knowledge_base(out)) == docs  # now a cache hit


def test_environment_indexes_documents_dir_end_to_end(facts_dir):
    out, ids = facts_dir
    env = get_environment(retrieval_variant="bm25", retrieval_kwargs={"documents_dir": str(out)})
    result = env.get_response(ToolCall(id="c1", name="KB_search", arguments={"query": MARKER})).content
    assert MARKER in result
    assert f"ID: {ids[0]}" in result
    assert get_knowledge_base().documents[ids[0]].content not in result
