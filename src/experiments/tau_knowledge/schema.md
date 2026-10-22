# banking_knowledge facts schema v1.1

Written 2026-11-07 from the document corpus only (`data/tau2/domains/banking_knowledge/documents/`), before reading any task file. Frozen for the 2x2 experiment in `PLAN-banking-facts.md`; any change here is a new corpus version. v1.1 (same day, before the full extraction) adds the table/list rule below after a pre-test rewrite flattened a merchant-exclusion list into `notes`; it comes from reading documents, not tasks.

## Rule

A facts document is the prose document rewritten as typed records. Same id, same title; only the content changes. Nothing is added and nothing is dropped: every number, amount, percentage, date, time window, product/account/card name, condition, exception and tool name (`name_1234`) in the prose appears in a record, verbatim. Omit a slot the prose says nothing about. Do not infer, summarize or normalize values ("$0.75 dollars" stays "$0.75 dollars"; "Green Account (checking)" stays as written).

Tables and lists keep their structure. A Markdown table (reason codes, tiers, fee schedules) stays a table and a grouped list (merchant exclusions by category, step sub-bullets) stays a grouped list, placed inside the slot that owns it (`exceptions`, `limits`, `fees`, `steps`, ...). Never flatten them into one-line prose or into `notes`.

## Record types

### Product
A product, account class, card, subscription tier or service the bank offers.
- name: official name as written in the prose
- category: checking | savings | credit card | debit card | business | payments | subscription | other
- tier: tier or class name when the product belongs to a family
- fees: each fee with its amount and condition
- apy: interest rates with their conditions
- cashback: reward rates per category
- eligibility: who qualifies (age, income, verification, account counts, history)
- limits: amounts, counts, time windows, caps
- promotions: bonus offers with their window (start/end or duration)
- notes: any other verbatim fact about the product

### Procedure
Something the agent or the customer does, step by step.
- name: short imperative (e.g. "Open personal checking account")
- trigger: the customer request or situation that starts it
- preconditions: everything that must be true before acting (verification, eligibility, holds, approvals)
- steps: numbered, in order, with verbatim tool names where the prose gives them
- requires_verification: yes | no | not stated
- tools: every `name_1234` tool the procedure uses and who calls it
- exceptions: when the procedure does not apply or is routed elsewhere (human transfer, denial, hold, next processing window)

### Tool
A discoverable tool the prose names (`name_1234`); one record per tool.
- name: exact token including the 4-digit suffix
- caller: agent | user | not stated (user tools are given to the customer; agent tools are unlocked and called by the agent)
- signature: arguments and their types or formats as the prose gives them
- unlock_required: yes | no | not stated
- when_to_use: the condition the prose attaches to using it

### Other
Verbatim text that fits none of the above (FAQ answers, definitions, general guidance). Keep sentences whole.

## Output format

Markdown only. One `###` header per record: `### Product: <name>`, `### Procedure: <name>`, `### Tool: <name>`, `### Other`. Slots as `- slot: value` lines; multi-value slots as nested `  - ` bullets; steps as nested `  1.` lines. No preamble, no closing remarks, no record for information that is not in the prose.
