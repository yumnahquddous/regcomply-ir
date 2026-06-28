# Information Retrieval Term Project Report

## 1. Project Title and Group Members
**Project Title:** Regulatory IR: RIRAG 2025 Shared Task (ADGM Regulatory Intelligence)
**Group Members:**
- Member 1: [Name] - [Roll Number]
- Member 2: [Name] - [Roll Number]
- Member 3: [Name] - [Roll Number]

## 2. Problem Statement
Regulatory compliance involves navigating massive and dense rulebooks to understand obligations accurately. The core retrieval need addressed by this project is automating the extraction of relevant regulatory passages and generating grounded answers to compliance-related questions. Our system supports answering both single-passage queries (where a single rule applies) and complex multi-passage queries (requiring integration of multiple rules) from the Abu Dhabi Global Market (ADGM) regulations.

## 3. Dataset Details
- **Source:** The ObliQA Dataset, a regulatory compliance QA dataset derived from ADGM regulations.
- **Size:** 27,869 annotated questions across train, dev, and test splits, encompassing 40 standardized structured regulatory documents.
- **Document Type:** Structured JSON and standardized text (.txt) rulebooks from various ADGM branches (e.g., AML, COBS, FUNDS, MKT).
- **Preprocessing Steps:**
  - Tokenization and lowercasing for lexical search.
  - Text normalization and cleaning of whitespace formatting.
  - Chunking rulebooks into passage segments using their native semantic passage IDs.

## 4. Methodology
Because regulatory queries vary widely (from exact keyword inquiries to conceptually nuanced questions), relying on a single Information Retrieval method is insufficient. 
We chose a **Hybrid Retrieval** method (System C) that combines:
- **Lexical Search (BM25):** Highly precise for exact matches on specific entities, rule names, or acronyms.
- **Semantic Search (Dense Embeddings):** Uses `multilingual-e5-base` sentence transformers to map user intent, matching queries based on conceptual meaning even if exact words are absent.
- **Score Normalization:** Both BM25 and Dense scores are normalized using Min-Max scaling. The scores are linearly combined using a hybrid formula (`alpha * dense + (1 - alpha) * bm25`) to rank the final passages.

## 5. System Workflow
The workflow generally follows a pipeline consisting of data ingestion, retrieval, and contextual generation (RAG):
1. **User Query Input:** Through the web UI or API endpoint.
2. **Hybrid Keyword/Semantic Search:** The backend computes both BM25 and embedding cosine similarities, normalizes, and ranks the passages.
3. **Context Construction:** The Top-K passages are combined into a context payload.
4. **Answer Generation (LLM):** An expert-instructed LLM (Gemini 1.5 Flash) synthesizes the finalized answer explicitly explicitly constrained to the provided context.
5. **Source Citation:** The UI directly displays exactly which documents contributed to the synthesis.

## 6. Evaluation Queries and Results
**Test Query 1 (Broad Keyword Search):** 
- *Query:* "What are the rules regarding anti-money laundering?"
- *Output:* Retrieves passage `3.2` from document `AML_VER09.211223.txt` (Hybrid Score: `0.8225`), correctly identifying broad AML standards. The LLM summarizes general obligations of an Authorized Person accurately and without hallucination.

**Test Query 2 (Specific Nuanced Question):**
- *Query:* "Can the ADGM provide clarity on the level of detail and documentation that should accompany a report of suspicious activity to ensure it meets regulatory standards?"
- *Output:* Successfully targets and retrieves passage `14.2.3.Guidance.10.` indicating guidance from EOCN regarding transaction reporting. System provides an integrated summary of relevant procedural actions matching regulatory thresholds.

## 7. Limitations
- **Multi-Hop Reasoning on Extensive Contexts:** The system occasionally struggles when a query spans three or more significantly disconnected source files, mostly due to exceeding token window constraints or drowning out the semantic similarity scores of secondary requirements.
- **Over-Reliance on Alpha Tuning:** The exact weighting parameter (alpha) between BM25 and Semantic search requires manual tuning. A static alpha can penalize specific exact-match rule searches.
- **Document Parsing Artifacts:** Tables and deeply nested hierarchical structures in the raw rulebooks occasionally flatten awkwardly, hindering precision in specific sub-clause retrievals.

## 8. Individual Contribution
- **Member 1:** Focused primarily on the implementation of the Retrieval Engine, structuring the data pipeline, extracting passages from ADGM texts, and tuning the Semantic Search (Cosine Similarity). 
- **Member 2:** Implemented the Lexical Search (BM25) and Min-Max scaling combination logic, integrated the Gemini RAG generator within FastAPI, and worked on final pipeline evaluation scripts.
- **Member 3:** Built the frontend client using Astro, handling document visualization minimaps, API data fetching, and preparing the presentation and report formatting, ensuring the tool is user-friendly and functional.
