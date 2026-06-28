# Presentation Slides Content: Regulatory IR

## Slide 1: Title Slide
**Title:** Regulatory Information Retrieval & Answer Generation
**Subtitle:** RIRAG 2025 Shared Task (ADGM Regulatory Intelligence)
**Presenters:** Adan Ayaz (Roll: 241447) & Yumnah Abdul Quddous (Roll: 241407)
**Course:** Information Retrieval 

---

## Slide 2: Problem Definition
- **The Challenge:** Navigating massive, dense regulatory rulebooks to locate specific compliance obligations. Traditional keyword searches are insufficient due to complex legal vernacular.
- **The Retrieval Need:** Automating the extraction of relevant regulatory passages and generating grounded answers for both simple and complex compliance scenarios.
- **Target Users:** Compliance officers, auditors, and legal analysts interacting with Abu Dhabi Global Market (ADGM) frameworks.

---

## Slide 3: Dataset Selection & Preparation
- **Dataset:** The *ObliQA Dataset* (from RegNLP 2025).
- **Size & Scope:** 27,869 QA pairs; 40 standardized ADGM regulatory text/JSON files.
- **Data Preprocessing:**
  - Tokenization & lowercasing for lexical indexing.
  - Chunking rulebooks conceptually by their native semantic passage IDs.
  - Pre-computing embeddings using `multilingual-e5-base` to optimize inference time.

---

## Slide 4: IR Technique Chosen
- **Why a Hybrid System?** Pure keyword matching fails on conceptual questions; pure semantic search misses hard entity names or specific rule numbers. 
- **System Composition:**
  - **Lexical Search:** BM25 (implemented via `rank-bm25`) captures exact terminology.
  - **Semantic Search:** Dense vector embeddings + Cosine Similarity align contextual meaning.
  - **Fusion:** Scores are normalized (Min-Max) and combined using an `alpha` parameter logic.

---

## Slide 5: System Workflow (RAG Architecture)
*(Instructor Tip: Verbally explain the flow while presenting this slide)*
1. **Input:** User sends a natural language query via the Astro Frontend.
2. **Retrieve:** FastAPI backend runs Hybrid Search on the index.
3. **Rank:** Top-K documents are scored and ranked.
4. **Augment:** Top passages are concatenated into a prompt.
5. **Generate:** The LLM (Gemini 1.5 Flash) answers using *only* the provided context and avoids external hallucination.
6. **Output:** System displays the answer and the original source files for validation.

---

## Slide 6: System Implementation Details
- **Backend:** Python + FastAPI for robust performance and async handling.
- **Frontend:** Astro framework (SSR/SSG built with UI cards and document previews).
- **Vector Model:** `intfloat/multilingual-e5-base` for dense representations.
- **Generation Model:** Google Gemini API configured with low temperature (`0.1`) for grounded, factual, un-creative answers.

---

## Slide 7: Evaluation of Retrieval Results 
- Evaluated on Subtask 1 metrics (Recall@10, MAP@10) and Subtask 2 metrics (RePASs).
- **Test Query 1 (Success):** 
  - *Query:* "What are the rules regarding anti-money laundering?" 
  - *Result:* Accurately returned Passage ID 3.2 (Hybrid Score: 0.82) containing primary AML directions.
- **Test Query 2 (Success):** 
  - *Query:* "Clarity on detail for suspicious activity reporting..." 
  - *Result:* Safely indexed specifically into guidance rule 14.2.3.

---

## Slide 8: Analyzing a Failure Case
- **Failure Scenario Example:** Multi-hop cross-referencing queries.
  - *Example query:* "How does the AML mandate differ under COBS framework versus normal banking environments?"
- **Why it failed:** 
  - Heavy divergence of vocabulary across different `.txt` files drops the BM25 alignment score significantly.
  - LLM receives context from one branch and completely ignores the other branch, leading to an incomplete answer.
- **Takeaway:** Cross-document synthesis remains a difficult retrieval hurdle compared to isolated factual retrieval.

---

## Slide 9: Demonstration Instructions (Placeholder)
- *(Live Demo portion: Switch screen to browser)*
- Run the local server.
- Execute Query 1: Single-passage retrieval.
- Execute Query 2: Multi-passage retrieval.
- Showcase document highlighting feature in UI.

---

## Slide 10: Conclusion & Next Steps
- **Recap:** A hybrid approach significantly boosts regulatory IR performance over standalone BM25.
- **Limitations:** Multi-hop reasoning and manual tuning of the fusion alpha parameter.
- **Future Work:** Implementing automated query-routing or dynamic alpha-weighting based on query type (e.g., higher BM25 weight if a specific rule number string is detected). 
- **Q&A Session Begins.**