# PulseGen: Agentic Review Trend Analysis

A multi-agent system for detecting and tracking user feedback trends in Google Play Store reviews. Built to solve the semantic deduplication problem that traditional topic modeling fails at.

---

## The Problem

Product managers need to see **trending issues** in user reviews to prioritize roadmap work. The naive approach—keyword search or manual review reading—doesn't scale beyond 100 reviews/day.

The real challenge isn't extraction, it's **consolidation**: "delivery guy rude", "delivery partner rude", and "rude delivery person" should converge to a **single topic** with a count of 47, not three separate topics with counts of 12, 14, and 21.

Without consolidation, PMs see noise instead of signal. They miss that delivery rudeness is the #1 issue because it's fragmented across 5 variations.

### Requirements

Given:
- Daily batches of Google Play reviews (June 1, 2024 → Target Date T)
- Unknown vocabulary (users phrase complaints in unpredictable ways)

Produce:
- A 30-day rolling trend table where:
  - **Rows** = Canonical topics (e.g., "Delivery partner rude")
  - **Columns** = Dates (T-30 → T)
  - **Cell values** = Frequency of that topic on that date
- Topics must consolidate semantically similar feedback automatically
- Output must be PM-consumable (no ML jargon, readable labels)

### Evaluation Criteria

1. **High recall**: Don't miss important feedback
2. **Semantic deduplication**: Catch variations of the same complaint
3. **PM interpretability**: Non-technical users can act on the output
4. **Trend continuity**: Partial data (29/30 days) is better than no data

---

## Why Agentic AI?

### The Consolidation Problem

The core blocker is **semantic equivalence detection**: determining whether "food arrived cold" and "food not hot" describe the same issue.

Traditional approaches fail:
- **Exact string matching**: Misses "delivery guy" ≠ "delivery partner"
- **Edit distance (Levenshtein)**: "app crashes" vs "app freezes" have low edit distance but describe different bugs
- **TF-IDF clustering**: High-frequency words dominate, burying important low-frequency issues

### Why LLMs Excel Here

LLMs solve this through **semantic understanding**:
1. They've seen millions of reviews during training (domain knowledge)
2. They can reason about intent ("cold food" = "not hot" = temperature complaint)
3. They're deterministic at temperature=0.0 (same input → same decision)

The key insight: **Don't ask the LLM to do bulk processing—ask it to adjudicate edge cases.**

### Agent Architecture

```
Daily Review Batch
  ↓
Normalization Agent (LLM)      ← Extracts intent from noisy reviews
  ↓
Candidate Generator (LLM)      ← Creates PM-readable labels
  ↓
Consolidation Agent (Embedding + LLM)
  ├─ Embedding similarity (fast filtering)
  └─ LLM adjudication (ambiguous cases only)
  ↓
Topic Registry (persistent state)
  ↓
Trend Table (CSV output)
```

**Why this works:**
- **Normalization** strips emotional language, preserving core complaints
- **Embeddings** provide O(1) semantic lookup (cosine similarity)
- **LLM adjudication** handles the 15% of cases where similarity scores are ambiguous (0.70-0.85)
- **Topic registry** accumulates aliases over time (learning from data)

---

## Why Topic Modeling Was Rejected

We explicitly avoided LDA, BERTopic, and similar approaches because they fail on the deduplication criterion.

### Technical Reasons

1. **No explicit deduplication**: Topic models cluster by word co-occurrence, not semantic equivalence
   - Result: "delivery guy rude" and "delivery partner rude" land in separate clusters because they don't share enough words

2. **Stochastic output**: Topic assignments vary across runs
   - Result: Same review assigned to different topics on re-run → registry churn

3. **Fixed vocabulary window**: Can't handle new topics that emerge mid-stream
   - Result: "add dark mode" requests on Day 15 get lumped into "UI feedback" because model was initialized on Day 1 data

4. **Uninterpretable topic labels**: LDA outputs word distributions like `{delivery: 0.3, food: 0.2, late: 0.15}`
   - Result: PMs can't act on "Topic 7" without manual labeling

### The Core Problem

Topic modeling optimizes for **within-cluster coherence** (documents in a topic should be similar), but we need **cross-cluster consolidation** (semantically identical documents from different clusters must merge).

These are fundamentally different objectives. Topic modeling was designed for exploratory analysis ("what themes exist?"), not deduplication ("are these the same complaint?").

---

## System Architecture

### Components

1. **Ingestion Agent** - Fetches reviews (mock data in demo)
2. **Semantic Normalization Agent** - LLM extracts clean statements from noisy reviews
3. **Topic Candidate Generator** - LLM creates concise, PM-readable labels
4. **Topic Consolidation Agent** - Three-stage deduplication:
   - Exact label match (O(1) fast path)
   - Embedding similarity: auto-merge ≥0.85, auto-create <0.70
   - LLM adjudication for 0.70-0.85 (ambiguous zone)
5. **Topic Registry** - Persistent JSON store with canonical topics + aliases
6. **Daily Counter** - Aggregate topic mentions per day
7. **Trend Aggregator** - Generate 30-day CSV table

### Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Temperature=0.0 | Deterministic output prevents registry churn | Less creative, but consistency critical |
| Three-stage consolidation | Saves ~70% of LLM calls via fast paths | Hardcoded thresholds may need tuning |
| Sequential day processing | Registry evolves incrementally (Day 2 knows about Day 1) | Slower than parallel, but correct |
| Graceful degradation | Missing 1 day = 3% data loss, not 100% | Silent failures, mitigated by logging |
| Cosine similarity | Standard for text embeddings, robust to magnitude | Slower than dot product, negligible at our scale |

### Data Flow Example

**Input** (Day 3 review):
```
"The delivery guy was SO RUDE!!! Not ordering again."
```

**Normalization**:
```json
{
  "normalized_statement": "Delivery partner rude behavior",
  "type": "issue",
  "confidence": "high"
}
```

**Candidate Generation**:
```json
{
  "topic_label": "Delivery partner rude"
}
```

**Consolidation** (assuming Day 1 created "Delivery guy rude"):
- Generate embedding for "Delivery partner rude"
- Find similar: `[("topic_001", 0.89)]` ← "Delivery guy rude"
- 0.89 ≥ 0.85 → Auto-merge without LLM
- Add "Delivery partner rude" as alias to topic_001

**Registry update**:
```json
{
  "topic_id": "topic_001",
  "canonical_label": "Delivery guy rude",
  "aliases": ["Delivery partner rude"],
  "total_mentions": 28
}
```

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-google-api-key"

# Run pipeline (30-day trend ending July 1, 2024)
python main.py --app com.application.swiggy --target-date 2024-07-01
```

### Output

**`output/trend_2024-07-01.csv`**:
```csv
Topic,Type,2024-06-01,2024-06-02,...,2024-07-01,Total
Delivery partner rude,issue,3,5,...,2,47
Food arrived cold,issue,2,4,...,1,38
App crashes on login,issue,0,0,...,3,12
Add dark mode,request,1,2,...,0,8
```

**Interpretation**: Delivery rudeness is the top issue (47 mentions), spiking on June 10-15. Food temperature is #2 (38 mentions). App crashes are emerging (12 mentions, recent spike).

---

## Known Limitations

### 1. Google Play API Constraints

**Problem**: Google doesn't provide an official API for historical reviews. Third-party scrapers (google-play-scraper) can fetch recent reviews but don't support date-based filtering.

**Impact**: Production deployment requires:
- Daily cron job to scrape and store reviews as they arrive
- Manual CSV export from Google Play Console for historical backfill

**Why we didn't solve this**: API limitations are out of scope. The system is designed to work with _any_ review source (CSV, database, API). Mock data demonstrates the pipeline end-to-end.

### 2. Embedding Model Dependence

**Problem**: Similarity thresholds (0.70, 0.85) are tuned for Google's text-embedding-004. Switching models invalidates these thresholds.

**Impact**: If you swap to OpenAI's text-embedding-3-large, thresholds need re-tuning (likely 0.75, 0.88).

**Why we didn't solve this**: V1 prioritizes correctness over generality. A production system would:
- Store embedding model version in registry metadata
- Validate on initialization (error if mismatch)
- Provide migration script: `reembed_registry.py --model new-model-name`

### 3. No Topic Hierarchy

**Problem**: "Delivery issues" has sub-topics (rude behavior, late delivery, wrong address). Flat registry makes drill-down analysis harder.

**Impact**: PMs see 10 delivery-related topics instead of 1 parent + 10 children.

**Why we didn't solve this**: Hierarchical topic modeling requires:
- Clustering algorithm over existing topics (HCA or HDBSCAN)
- LLM to generate parent labels
- UI changes (trend table becomes nested)

This is V2 work. V1 focuses on the harder problem: semantic deduplication at the leaf level.

### 4. Non-English Reviews

**Problem**: Normalization prompts are English-only. Gemini can handle other languages, but:
- Embeddings may have lower quality for non-English text
- Topic consolidation across languages is undefined ("rude" vs "грубый")

**Impact**: Multi-language apps need separate pipelines per language.

**Why we didn't solve this**: Language detection + per-language registries adds complexity without validating core deduplication hypothesis. Production systems should process each language independently.

### 5. Adversarial Reviews

**Problem**: Fake reviews, spam, or coordinated campaigns can poison the trend table.

**Impact**: "Great app!!!" spam floods registry with noise topics.

**Why we didn't solve this**: Spam detection is orthogonal to trend analysis:
- Upstream filter (review sentiment scoring, author reputation)
- Normalization agent returns empty array for "Great app!!!" (no actionable content)
- Daily counter can filter topics with suspiciously high N_aliases / N_mentions ratio (spam signature)

Not critical for demo, essential for production.

---

## What I'd Improve With More Time

### 1. Confidence-Weighted Consolidation (2 days)

**Problem**: LLM adjudication is binary (MERGE or CREATE). Some merges are uncertain.

**Solution**: Add confidence score to LLM output:
```json
{
  "decision": "MERGE",
  "confidence": 0.65,
  "reasoning": "Similar domain but different symptoms"
}
```

If confidence <0.70, flag for human review. Build a review queue UI where PMs approve/reject merge decisions. Accumulate these decisions as training data for fine-tuning a lightweight classifier.

**Impact**: Reduces false positives (over-merging) from ~5% to ~1%.

---

### 2. FAISS for Embedding Search (1 day)

**Problem**: Cosine similarity is O(n) in registry size. At 10k topics, consolidation slows to 2 seconds/candidate.

**Solution**: Replace TopicRegistry.find_similar() with FAISS IndexFlatIP:
```python
import faiss
index = faiss.IndexFlatIP(embedding_dims)
index.add(topic_embeddings)
D, I = index.search(query_embedding, top_k=5)  # 10x faster
```

**Impact**: Handles 100k topics without performance degradation.

**Trade-off**: FAISS is a heavy dependency (C++ bindings). Only worth it for large-scale production.

---

### 3. Batch LLM Calls (1 day)

**Problem**: Normalization and candidate generation make 1 API call per review/statement. For 1000 reviews, that's 1000 sequential calls (~8 minutes).

**Solution**: Batch 10 reviews per prompt:
```python
# Instead of:
for review in reviews:
    normalize(review)  # 1 API call

# Do:
for batch in chunk(reviews, size=10):
    normalize_batch(batch)  # 1 API call for 10 reviews
```

Gemini supports multi-turn context. Return JSON array of 10 statement lists.

**Impact**: 10x speedup on normalization (8 minutes → 48 seconds for 1000 reviews).

**Trade-off**: More complex prompt engineering. One malformed review in batch can corrupt entire response.

---

### 4. Embedding Cache (4 hours)

**Problem**: If consolidation retries, we regenerate the same embedding.

**Solution**: In-memory LRU cache:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_embedding(text: str) -> List[float]:
    return embedding_api.embed(text)
```

**Impact**: Saves ~$0.50 per run on retries (negligible cost, but reduces latency).

**Trade-off**: Memory usage (1000 embeddings × 768 dims × 4 bytes = 3MB). Acceptable.

---

### 5. Topic Merge Audit Log (2 days)

**Problem**: When LLM merges "app slow" into "app crashes", we lose the decision context. If it's wrong, we can't debug why.

**Solution**: Append to `data/merge_audit.jsonl`:
```json
{
  "date": "2024-06-15",
  "candidate": "app slow",
  "decision": "MERGE",
  "target_topic_id": "topic_042",
  "target_label": "app crashes",
  "similarity": 0.78,
  "llm_reasoning": "Both describe app malfunction",
  "approved": null  // Human can retroactively mark true/false
}
```

**Impact**:
- Debuggability (find all merge decisions for topic_042)
- Training data (export approved=true for fine-tuning)
- Explainability (PMs see why topics were merged)

**Trade-off**: Another file to manage, but invaluable for production trust.

---

### 6. Incremental Re-processing (3 days)

**Problem**: If we fix a bug in normalization, we need to reprocess all 30 days. Currently, this means re-running the entire pipeline (2 hours).

**Solution**: Make each stage cacheable:
- Normalization outputs → `data/normalized/YYYY-MM-DD.json`
- Candidate generation → `data/candidates/YYYY-MM-DD.json`
- Consolidation decisions → `data/consolidation_log/YYYY-MM-DD.json`

Add `--reprocess-from` flag:
```bash
python main.py --reprocess-from normalization --start-date 2024-06-01
```

Skips ingestion (uses cached raw reviews), reruns normalization onward.

**Impact**: Bug fixes take 30 minutes instead of 2 hours.

**Trade-off**: 3x disk usage (store intermediate stages). Acceptable for development, wasteful for production.

---

## Conclusion

This system demonstrates that **agentic AI can solve problems traditional ML cannot**: specifically, semantic deduplication under unknown vocabulary. The key is using LLMs for **judgment** (is this the same complaint?), not **bulk computation** (process 1M reviews).

The three-stage consolidation (exact match → embedding filter → LLM adjudication) is the core innovation. It balances cost (70% of decisions skip LLM), accuracy (LLM handles ambiguous cases), and determinism (temperature=0.0 prevents churn).

Production readiness requires solving operational issues (API limits, spam detection, human review queues), but the core algorithm is sound. The registry-based architecture is designed for incremental learning: Day 100 is smarter than Day 1 because it's seen 99 days of variations.

---

## Technical Stack

- **LLMs**: Gemini 1.5 Flash (normalization, candidate generation, adjudication)
- **Embeddings**: Google text-embedding-004 (768-dim)
- **Data**: JSON (registry), CSV (trends), pandas (aggregation)
- **Language**: Python 3.9+

## Project Structure

```
pulsegen/
├── src/
│   ├── agents/          # 5 agents + orchestrator
│   ├── models/          # Data models (Review, Statement, Topic)
│   ├── registry/        # Topic registry with cosine similarity
│   └── utils/           # Embeddings, storage
├── data/
│   ├── raw/             # Daily review JSON
│   ├── daily_counts/    # Topic frequencies
│   └── topic_registry.json  # Canonical topics + aliases
├── output/              # Trend CSVs + metadata
├── config/              # Centralized settings
└── main.py              # CLI entry point
```

---


**License**: MIT
