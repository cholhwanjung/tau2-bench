from experiments.tau_knowledge.make_facts import TOOL_RE, other_share, validate_facts

PROSE = (
    "## Eligibility\n1. The customer is at least 18 years old.\n2. No more than 4 checking accounts.\n"
    "3. Fee is $1,000 or 2.5% after 11/14/2025.\n\n## Procedure\n1. Verify identity.\n2. Use open_bank_account_4821 to open.\n"
    "3. Transfer with transfer_funds_between_bank_accounts_7291."
)


def test_tool_regex_matches_corpus_style_names_only():
    assert TOOL_RE.findall("use open_bank_account_4821 and initial_transfer_to_human_agent_0218") == [
        "open_bank_account_4821",
        "initial_transfer_to_human_agent_0218",
    ]
    assert TOOL_RE.findall("task_001 doc_bank_accounts_(general)_001 version_12345") == []


def test_validate_passes_when_every_token_survives():
    facts = (
        "### Procedure: Open personal checking account\n- preconditions:\n  - at least 18 years old\n"
        "  - no more than 4 checking accounts\n  - fee $1,000 or 2.5% after 11/14/2025\n"
        "- steps:\n  1. Verify identity\n  2. open_bank_account_4821\n  3. transfer_funds_between_bank_accounts_7291\n"
    )
    assert validate_facts(PROSE, facts) == {"ok": True, "missing_tools": [], "missing_numbers": []}


def test_validate_reports_missing_tool_and_number():
    facts = "### Procedure: Open\n- steps:\n  1. open_bank_account_4821\n- fee: $1000 or 2.5% after 11/14/2025\n- age: 18\n- max accounts: 4"
    check = validate_facts(PROSE, facts)
    assert check["ok"] is False
    assert check["missing_tools"] == ["transfer_funds_between_bank_accounts_7291"]
    assert check["missing_numbers"] == ["1,000"]  # "1000" is not the verbatim token


def test_list_markers_are_not_numeric_facts():
    # Fewer numbered steps in the facts must not count as dropped numbers.
    assert validate_facts("1. a\n2. b\n3. c\n4. pay $5", "### Other\n- a, b, c, pay $5")["ok"]
    # Corpus style "Steps: 1) ..., 2) ..." inline and "3) " at line start.
    prose = "Eligibility: 1) verified, 2) at least 14 days. Steps:\n1) verify\n2) check\n3) open"
    assert validate_facts(prose, "- preconditions: verified; at least 14 days\n- steps: verify, check, open")["ok"]
    # A real small number is still required.
    assert validate_facts("Hold for 2 days.", "- hold: two days")["missing_numbers"] == ["2"]


def test_other_share_counts_only_other_blocks():
    facts = "### Product: X\n- fees: $1\n### Other\n- verbatim sentence\n### Tool: t_1234\n- caller: agent\n"
    share = other_share(facts)
    assert 0.0 < share < 0.5
    assert other_share("### Other\n- all of it") == 1.0
    assert other_share("### Product: X\n- fees: $1") == 0.0


def test_title_corpus_keeps_ids_and_prepends_title(tmp_path):
    import json

    from experiments.tau_knowledge.make_title_corpus import build, title_content

    src = tmp_path / "src"
    src.mkdir()
    (src / "doc_a.json").write_text(json.dumps({"id": "doc_a", "title": "Card X: Fees", "content": "Annual fee $95."}))
    assert build(src, tmp_path / "out") == 1
    out = json.loads((tmp_path / "out" / "doc_a.json").read_text())
    assert out["id"] == "doc_a" and out["title"] == "Card X: Fees"
    assert out["content"] == title_content({"title": "Card X: Fees", "content": "Annual fee $95."}) == "# Card X: Fees\n\nAnnual fee $95."


def test_required_doc_ids_exist_in_corpus():
    from experiments.tau_knowledge.make_facts import required_doc_ids
    from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR

    ids = required_doc_ids()
    assert len(ids) == 218  # distinct documents some task requires (43% of the corpus text)
    assert all((KNOWLEDGE_DOCUMENTS_DIR / f"{i}.json").exists() for i in ids)
