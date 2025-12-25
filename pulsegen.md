# **Review Trend Analysis Agent (Agentic AI)**

## **1\. Purpose of This Document**

This document is a **build-ready technical and product specification** for creating an **Agentic AI system** that ingests daily Google Play Store reviews and produces a **30-day rolling trend analysis table** of issues, requests, and feedback.

This is written so that:

* A developer can implement directly without ambiguity  
* Architectural decisions are explicit  
* Trade-offs are documented  
* Evaluation criteria of Pulsegen are directly satisfied

This document prioritizes **correctness, high recall, semantic consolidation, and PM usability** over academic topic modeling.

---

## **2\. Problem Definition**

### **Input**

* App identifier (Google Play Store app link or package name)  
* Target date **T**  
* Daily batches of reviews from **June 1, 2024 → T**

Each day is treated as an independent batch.

### **Output**

A **trend table** where:

* Rows \= Canonical topics (issues / requests / feedback)  
* Columns \= Dates from **T-30 → T**  
* Cell value \= Frequency of topic occurrence on that date

The output must:

* Consolidate semantically similar feedback into a single topic  
* Detect **new, evolving topics** automatically  
* Avoid topic fragmentation

---

## **3\. Non-Goals (Explicitly Out of Scope)**

* Academic topic modeling (LDA, BERTopic, TopicBERT)  
* Sentiment-only analysis  
* Perfect linguistic accuracy  
* UI / dashboard (CSV / table output is sufficient)

---

## **4\. Core Design Principles**

1. **High Recall First**  
   Missing a topic is worse than temporarily over-grouping  
2. **Semantic Canonicalization**  
   Similar meanings must converge to a single topic  
3. **Agentic Decision Making**  
   LLMs are used for judgment, not bulk computation  
4. **Incremental Learning**  
   Topic registry evolves over time  
5. **PM-Consumable Output**  
   Output must be interpretable by non-technical users

---

## **5\. High-Level Architecture**

Daily Review Batch (Date D)  
        ↓  
Ingestion Agent  
        ↓  
Semantic Normalization Agent  
        ↓  
Topic Candidate Generator  
        ↓  
Topic Consolidation Agent  
        ↓  
Topic Registry (Canonical Topics)  
        ↓  
Daily Topic Counter  
        ↓  
30-Day Trend Aggregator  
        ↓  
Trend Table Output (CSV)  
---

## **6\. Component-Level Specifications**

### **6.1 Ingestion Agent**

**Responsibility**

* Fetch reviews for a specific app and date

**Inputs**

* App package name  
* Date D

**Outputs**

* List of raw reviews

**Implementation Notes**

* Use google-play-scraper or equivalent  
* If historical scraping is unreliable, mock historical batches with documentation  
* Store raw reviews in /data/raw/YYYY-MM-DD.json

---

### **6.2 Semantic Normalization Agent**

**Responsibility**

Convert raw review text into **clean, intent-focused statements** suitable for semantic comparison.

**Input**

* Raw review text

**Output**

{  
  "normalized\_statement": "Delivery partner rude behavior",  
  "type": "issue" | "request" | "feedback"  
}

**LLM Prompt Behavior**

* Remove emotional language  
* Remove app-specific fluff  
* Preserve core complaint or request  
* One review may generate **multiple statements**

**Why This Matters**

This step dramatically improves downstream topic consolidation accuracy.

---

### **6.3 Topic Candidate Generator**

**Responsibility**

* Generate short, human-readable topic labels from normalized statements

**Example**

Input:

“Delivery partner rude behavior”

Output:

“Delivery partner rude”

This step ensures topics are PM-readable.

---

### **6.4 Topic Registry (Ontology-Lite)**

**Purpose**

Acts as the **single source of truth** for all discovered topics.

**Schema**

{  
  "topic\_id": "uuid",  
  "canonical\_label": "Delivery partner rude",  
  "aliases": \[  
    "delivery guy rude",  
    "delivery person impolite"  
  \],  
  "embedding": \[0.123, 0.456, ...\],  
  "created\_on": "2024-06-03"  
}

Stored in /data/topic\_registry.json

---

### **6.5 Topic Consolidation Agent (CRITICAL)**

**Responsibility**

Decide whether a new topic candidate:

* Maps to an existing topic  
* Or represents a genuinely new topic

**Process**

1. Generate embedding for new candidate  
2. Compute similarity with all existing topics  
3. Retrieve top-K similar topics  
4. Ask LLM:  
   * Should this candidate MERGE or CREATE?

**Decision Criteria**

* Semantic equivalence  
* PM interpretability  
* Trend usefulness

**Output**

* Existing topic ID (merge)  
* OR new topic creation

This agent directly solves the “delivery guy rude” problem stated in the assignment.

---

### **6.6 Daily Topic Counter**

**Responsibility**

* Count occurrences of canonical topics for a single day

**Output**

{  
  "date": "2024-06-15",  
  "Delivery partner rude": 12,  
  "Food stale": 7  
}

Stored in /data/daily\_counts/YYYY-MM-DD.json

---

### **6.7 Trend Aggregator**

**Responsibility**

* For date T, aggregate counts from T-30 → T

**Output**

* CSV / table exactly matching assignment format

---

## **7\. Agent Coordination Logic**

Agents run sequentially per day.

Pseudo-flow:

for day in date\_range:  
  reviews \= ingest(day)  
  statements \= normalize(reviews)  
  topics \= generate\_candidates(statements)  
  canonical\_topics \= consolidate(topics)  
  count\_daily(canonical\_topics)

if day \== T:  
  generate\_trend\_table(T)  
---

## **8\. Failure Modes & Mitigations**

| Risk | Mitigation |
| ----- | ----- |
| Topic explosion | High similarity threshold \+ LLM adjudication |
| Over-merging | Human-readable labels \+ alias tracking |
| Missing trends | High recall normalization |
| Noisy data | Ignore reviews \< N characters |

---

## **9\. Evaluation Alignment (Pulsegen)**

| Requirement | How It’s Addressed |
| ----- | ----- |
| Agentic AI | Multi-agent decision flow |
| High recall | Normalization \+ embedding recall |
| Deduplication | Registry \+ LLM merge logic |
| PM usability | Canonical labels \+ trend table |

---

## **10\. Deliverables Checklist**

* /src agent code  
* /data/raw daily reviews  
* /data/topic\_registry.json  
* /output/trend\_T.csv  
* README.md (architecture \+ rationale)  
* Demo video

---

## **11\. Explicit Design Justification**

This system avoids traditional topic modeling due to:

* Semantic drift  
* Poor deduplication  
* Low PM trust

Agentic LLM-driven consolidation provides **controlled reasoning**, which is essential for product analytics.

---

## **12\. Success Criteria**

The system is successful if:

* Semantically identical feedback converges into one topic  
* New issues are detected automatically  
* Trend table clearly shows rising / falling topics  
* Output is understandable by a PM without explanation

---

END OF DOCUMENT

