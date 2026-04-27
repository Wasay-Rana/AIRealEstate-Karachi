# Graph-Enhanced RAG Agent — Karachi Real Estate Intelligence

A production-ready AI system combining **LightRAG** knowledge graphs, **Pinecone** vector search, and **Claude** reasoning to answer complex Karachi property questions — possession status, price history, tax calculations, litigation checks, and multi-hop investment analysis.

## Why This Matters

Traditional search tools fail on questions like:

> *"Which DCK sectors have given possession and what are the main delay reasons for those that haven't?"*

or

> *"Compare Malir City Phase 1 vs Saadi Town price trends and explain which is better value investment."*

These questions require synthesizing possession records, price data, legal framework, and investment context across multiple sources. No single document or keyword search contains the full answer.

**Graph-Enhanced RAG** solves this by:
1. Building a knowledge graph linking Societies → Sectors/Precincts → Possession Status → Price Index → Legal Flags
2. Traversing the graph for multi-hop reasoning (plot → society → tax rule → duty calculation)
3. Combining graph results with vector search for complete coverage
4. Generating grounded, cited answers using Claude

---

## Architecture

```
Query
  │
  ├─ QueryRewriter (Claude) ──── domain-specific expansion
  │
  ├─ ComplexityDetector ──────── fast | balanced | deep
  │      fast     = Pinecone only           (simple lookups)
  │      balanced = Pinecone + BM25         (tax/price queries)
  │      deep     = Pinecone + LightRAG     (multi-hop: society → sector → possession → litigation)
  │
  ├─ Fan-out retrievers [asyncio.gather]
  │
  ├─ ResultMerger → RRF deduplication
  │
  ├─ CrossEncoderReranker
  │
  ├─ ContextCompressor (6000 token budget)
  │
  └─ Claude (prompt caching) → grounded answer + citations
```

**Dual Storage:**
- **LightRAG** → knowledge graph of Karachi property entities and relationships
- **Pinecone** → dense vector search over 5 domain knowledge documents

---

## Knowledge Base

Five domain-specific documents ingested at startup:

| Document | Content |
|----------|---------|
| `dha_city_karachi_overview.txt` | DCK sectors, possession status 2024–2025, prices, transfer process, FBR rates |
| `bahria_town_karachi_guide.txt` | BTK precincts 1–45, Phase 2 extension, price trends, 2024 correction event |
| `malir_cantonment_projects.txt` | Malir City + Malir Town Residencia ballots, MCDA rules, 2024 Phase 3 ballot |
| `sindh_taxes_legal_framework.txt` | Sindh Finance Act 2024, stamp duty slabs, CVT, FBR July 2024 revision, SBCA digital NOC |
| `karachi_property_price_index.txt` | Quarterly price-per-sqyd data 2023 Q1–2025 Q1 for DCK, BTK, Malir City, Saadi Town |
| `karachi_property_buying_guide.txt` | Due diligence checklist, transfer steps, GPA rules, Zameen/Graana platform guide |

### LightRAG Entity Types (Real Estate Domain)

The knowledge graph extracts and links these entity types:

```
Society, Sector, Precinct, Block, PlotFile, FileStatus,
PossessionRecord, DutyRecord, PriceIndex, LitigationFlag,
DigitalNOC, RegulatoryAuthority, TaxRate, Developer,
LegalCase, Person, Organization
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| API Server | FastAPI + Uvicorn |
| Graph Engine | LightRAG (lightrag-hku) |
| Vector DB | Pinecone v5 (Serverless) |
| LLM | Claude claude-sonnet-4-6 |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Keyword Search | BM25 (rank_bm25) |

---

## Setup

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

### 2. Configure

```bash
cp .env.example .env
# Add: ANTHROPIC_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY
```

### 3. Run

```bash
make run        # http://localhost:8000
make ingest     # loads 6 Karachi real estate documents
make queries    # runs 3 example multi-hop queries
make test       # unit tests (no API keys needed)
```

---

## API Reference

### RAG Query Endpoint

```
POST /api/v1/query
```

```json
{
  "query": "Which Bahria Town precincts have possession and which are still booking-only in 2025?",
  "mode": "auto",
  "namespace": "karachi_property",
  "rewrite_query": true
}
```

**Retrieval modes:**
- `auto` — detects complexity; complex possession/litigation/multi-society queries → `deep`
- `fast` — vector search only (simple definitions)
- `balanced` — Pinecone + BM25 (price/tax lookups)
- `deep` — Pinecone + LightRAG graph (multi-hop: society → sector → possession chain)

---

### Property-Specific Endpoints

#### Possession Status
```
GET /api/v1/properties/possession-status?society=dha_city_karachi&sector_or_precinct=14
```

```json
{
  "society": "Dha City Karachi",
  "sector_or_precinct": "14",
  "total_plots": 1400,
  "possession_given": 1200,
  "possession_pending": 200,
  "pct_complete": 85.7,
  "pending_reasons": ["Sui gas not yet available", "Hospital not yet operational"],
  "as_of": "Q1 2025"
}
```

#### Price History
```
GET /api/v1/properties/price-history?society=bahria_town_karachi&sector_or_precinct=precinct_19&plot_size_sqyd=250&from_quarter=2023-Q1&to_quarter=2025-Q1
```

```json
{
  "society": "bahria_town_karachi",
  "latest_price_sqyd": 35500,
  "change_pct": 18.3,
  "data_points": [
    {"quarter": "2023-Q1", "price_per_sqyd_pkr": 30000, ...},
    {"quarter": "2024-Q2", "price_per_sqyd_pkr": 30000, ...},
    {"quarter": "2025-Q1", "price_per_sqyd_pkr": 35500, ...}
  ]
}
```

#### Litigation Check
```
GET /api/v1/properties/litigation-check?file_number=DCK-S09-0121
```

```json
{
  "file_number": "DCK-S09-0121",
  "has_litigation": true,
  "safe_to_transfer": false,
  "restriction_summary": "LITIGATION ALERT: 1 active flag(s). Restrictions: no transfer",
  "flags": [{"court_name": "SHC", "case_number": "SHC-2023-1045", ...}]
}
```

#### Stamp Duty Calculator
```
POST /api/v1/properties/duty-calculator
```

```json
{
  "society": "bahria_town_karachi",
  "declared_value_pkr": 18000000,
  "fbr_valuation_pkr": 7500000,
  "property_type": "residential",
  "buyer_is_filer": true,
  "seller_is_filer": true
}
```

```json
{
  "taxable_value_pkr": 18000000,
  "stamp_duty_rate_pct": 4.0,
  "stamp_duty_pkr": 720000,
  "cvt_rate_pct": 2.0,
  "cvt_pkr": 360000,
  "wht_seller_pkr": 180000,
  "wht_buyer_pkr": 180000,
  "total_tax_pkr": 1440000,
  "finance_act_applied": 2024
}
```

---

## Example Multi-Hop Queries

| # | Query | Mode | Documents |
|---|-------|------|-----------|
| Q1 | "Which DCK sectors have given possession and what are the delay reasons?" | deep | DCK overview + buying guide |
| Q2 | "What stamp duty on BTK Precinct 35 at PKR 1.8 crore under Finance Act 2024?" | balanced | Tax framework + BTK guide |
| Q3 | "Compare Malir City Phase 1 vs Saadi Town price trend 2023–2025: which is better value?" | deep | Price index + Malir guide + buying guide |

---

## File Status Reference

| Status | Meaning |
|--------|---------|
| `allotted` | Original allotment confirmed after ballot |
| `transferred` | File bought/sold; records updated at DHA/Bahria/MCDA |
| `on_hold` | Incomplete documentation or pending dues |
| `disputed` | Two+ parties claim same plot |
| `cancelled` | Allotment cancelled (non-payment); recoverable within 60 days |
| `possession_given` | Physical possession handed over; cleared for construction |
| `under_litigation` | Court stay — no transfer or possession until resolved |

---

## Tax Quick Reference (Sindh Finance Act 2024)

| Transaction Value | Stamp Duty | CVT (Residential) | CVT (Cantonment) |
|------------------|-----------|-------------------|-----------------|
| Up to PKR 5M | 3% | 2% | 1% |
| PKR 5M – 20M | 4% | 2% | 1% |
| Above PKR 20M | 5% | 2% | 1% |

Plus: WHT 236C/236K = 1% (filer) or 2% (non-filer) on each party.

---

## Known Constraints

- **Price data is static** in this demo. In production, wire to Zameen API or scheduled CSV ingestion.
- **Possession records** are sourced from the knowledge corpus. In production, replace with live DHA/Bahria/MCDA API integration.
- **Litigation DB** uses demo records. In production, integrate SHC cause-list scraper or legal data provider.
- **BM25 in-memory**: suitable up to ~50K chunks; replace with Elasticsearch for production scale.
- **LightRAG NanoVectorDB**: suitable for this corpus; upgrade to Qdrant for larger datasets.
