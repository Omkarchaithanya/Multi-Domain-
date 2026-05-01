# Multi-Domain Support Triage Agent

Automated support ticket triage for three domains — **HackerRank**, **Claude**, and **Visa** — using hybrid retrieval, LLM response generation, and a self-verification grounding loop.

---

## Setup

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

**Required packages:**
- `anthropic>=0.40.0` — LLM response generation and grounding verification
- `rank-bm25>=0.2.2` — BM25 keyword retrieval
- `sentence-transformers>=2.7.0` — Semantic embedding retrieval
- `numpy>=1.26.0`
- `requests>=2.31.0` + `beautifulsoup4>=4.12.0` — Web scraper

### 2. (Optional) Set your Anthropic API key

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# macOS/Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

Without an API key the system falls back to extractive responses from the corpus — all routing, escalation, and output schema logic still works.

### 3. (Optional) Refresh the corpus

The `data/` directory contains pre-scraped JSON chunks. To re-scrape from live support sites:

```bash
python code/scraper.py
```

This crawls up to 60 pages per domain (HackerRank, Claude, Visa support sites) and writes chunked JSON files to `data/{domain}/`.

### 4. Run the triage pipeline

```bash
python code/main.py
```

Reads `support_tickets/support_tickets/support_tickets.csv`, writes results to `support_tickets/support_tickets/output.csv`.

---

## Output Schema

| Column | Description |
|---|---|
| `ticket_id` | Original ticket identifier |
| `status` | `replied` or `escalated` |
| `product_area` | Domain: `hackerrank`, `claude`, or `visa` |
| `request_type` | `product_issue`, `bug`, `feature_request`, or `invalid` |
| `response` | Customer-facing reply |
| `justification` | Why the ticket was replied or escalated, with top source |

---

## Architecture

### Pipeline (left to right)

```
Ticket text
  → Intent Decomposer        (multi-intent extraction)
  → Tone Detector            (frustrated / urgent / confused / neutral)
  → Domain Router            (hackerrank / claude / visa)
  → Request Type Classifier  (billing / bug_report / feature_request / …)
  → Adversarial Check        (OOS / injection detection)
  → Escalation Decision      (rules-based: fraud, identity, billing disputes)
  → Hybrid Retriever         (BM25 + semantic + diversity penalty)
  → Response Generator       (LLM with grounding loop, or extractive fallback)
  → Output CSV
```

### Key modules

**`code/corpus_indexer.py`**
Loads `data/{domain}/*.json` (scraped pages) and builds per-domain BM25 and semantic indexes. Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings; falls back to a deterministic hashing encoder when the model is unavailable.

**`code/scraper.py`**
Crawls HackerRank, Claude, and Visa support sites (BFS, depth 2, max 60 pages each). Saves `{source, domain, text, url}` JSON chunks to `data/`.

**`code/retriever.py`**
Hybrid retrieval: 52% BM25 + 43% semantic + 5% rank bonus. Applies a 35% diversity penalty to chunks already used as the top result for a previous ticket in the same run. Falls back to query expansion if the top score is below 0.22.

**`code/response_generator.py`**
- **LLM path** (API key present): calls `claude-sonnet-4-20250514` with a tone-aware system prompt. A second LLM call verifies every factual claim in the response against the retrieved documentation (grounding score + verdict). Responses scoring below 0.7 or flagged `escalate` are replaced with the standard fallback. A third LLM call anticipates and answers the most likely follow-up question.
- **Extractive fallback** (no API key): selects the best chunk that starts at a sentence boundary, deduplicates sentences via exact + near-duplicate token overlap, and strips navigation boilerplate and API dump artifacts.
- Every response ends with *"Let us know if you need further help."*

**`code/escalation.py`**
Rules-based escalation triggers: identity verification lockout, billing disputes requiring account lookup, fraud/chargeback, permissions issues. Returns a reason string used in the `justification` field.

**`code/domain_router.py`**
Keyword + TF-IDF routing across the three domains with confidence scoring.

**`code/intent_decomposer.py`**
Splits compound tickets into sub-intents (e.g. "My account is locked AND I was double-charged" → two separate intent signals fed into routing).

**`code/tone_detector.py`**
Classifies ticket tone as `frustrated`, `urgent`, `confused`, `angry`, or `neutral`. The LLM prompt is adapted per tone.

**`code/logger.py`**
Structured JSON event logger. Writes `log.txt` alongside `output.csv`.

---

## Innovations

1. **Multi-Intent Decomposition** — handles tickets that contain more than one request, routing each sub-intent independently before merging.

2. **Hybrid Retrieval with Diversity Penalty** — BM25 + semantic fusion with a run-level penalty so multiple tickets never get identical top chunks.

3. **Self-Verification Grounding Loop** — a second LLM call fact-checks every claim in the generated response against the retrieved docs. Ungrounded responses are revised or escalated rather than sent.

4. **Proactive Follow-up Anticipation** — a third LLM call predicts the customer's next question and appends a pre-emptive answer when documentation supports it.

5. **Tone-Adaptive Response Generation** — the system prompt adjusts based on detected customer tone (e.g. leads with empathy for `frustrated`, skips preamble for `urgent`).

---

## Project Structure

```
├── code/
│   ├── main.py               # Entry point — runs the full pipeline
│   ├── corpus_indexer.py     # Builds BM25 + semantic indexes from data/
│   ├── scraper.py            # Crawls support sites → data/{domain}/*.json
│   ├── retriever.py          # Hybrid BM25 + semantic retrieval
│   ├── response_generator.py # LLM response + grounding loop + extractive fallback
│   ├── domain_router.py      # Routes tickets to hackerrank / claude / visa
│   ├── escalation.py         # Escalation rules + request-type classification
│   ├── intent_decomposer.py  # Multi-intent decomposition
│   ├── tone_detector.py      # Customer tone detection
│   ├── logger.py             # Structured JSON event logger
│   └── requirements.txt      # Python dependencies
├── data/
│   ├── hackerrank/           # Scraped HackerRank support JSON chunks
│   ├── claude/               # Scraped Claude support JSON chunks
│   └── visa/                 # Scraped Visa support JSON chunks
└── support_tickets/
    └── support_tickets/
        ├── support_tickets.csv       # Input tickets
        ├── sample_support_tickets.csv
        ├── output.csv                # Generated output
        └── log.txt                   # Structured run log
```
