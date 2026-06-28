# Potential Viva / Technical Defense Q&A

This document prepares the group for potential questions the instructor may ask during the Viva and live presentation. 

### 1. General Information Retrieval (IR) Questions
**Q: Why did you choose BM25 over basic TF-IDF?**
*A:* TF-IDF doesn't account for document length normalization inherently and struggles with term frequency saturation. BM25 penalizes overly long documents and uses saturation curves, making it much more effective for retrieving specific sub-passages from legal documents.

**Q: Explain how Cosine Similarity works in your Semantic Search.**
*A:* The `SentenceTransformer` model converts both the text passages and the user's query into high-dimensional numerical vectors (embeddings). Cosine similarity measures the cosine of the angle between these two vectors. If the vectors point in the same direction (meaning the semantic context is highly related), the score approaches 1.

**Q: How does your Hybrid Search formally combine scores?**
*A:* Since BM25 yields unbound scores (e.g., 10 to 50+) and cosine similarity gives normalized scores between 0 and 1, we first apply Min-Max scaling to the BM25 scores to bind them between 0 and 1. We then use a weighted sum formula: `final_score = (alpha * dense_score) + ((1 - alpha) * bm25_score)`. 

### 2. Dataset and Preprocessing
**Q: What specific preprocessing did you perform on the dataset?**
*A:* We loaded the passage data using Pandas, performed standard tokenization using `nltk`, and lowercased the text. For dense embeddings, we passed the chunked regulatory passages into `multilingual-e5-base` and stored the output mappings as an `.npy` file. 

**Q: How did you handle document chunking?**
*A:* We relied primarily on the innate structure of the rulebooks (which provide semantic passage IDs like `14.2.3.Guidance.10`). Keeping conceptual rule boundaries intact is vital for legal domains so we don't accidentally split a rule in half.

### 3. Implementation and System Workflow
**Q: What is a Retriever-Augmented Generation (RAG) system?**
*A:* RAG pairs an Information Retrieval mechanism (our hybrid search) with a generative LLM (Gemini). The IR system first finds the absolute best documents for a query. We inject these text chunks into the prompt of the generator. This forces the generator to answer based *only* on the factual text we provided instead of hallucinating answers.

**Q: How is the system run, and what dependencies are required?**
*A:* We use a microservice approach. The backend is a Python FastAPI application requiring libraries like `sentence-transformers`, `rank-bm25`, and `scikit-learn` (run via `uvicorn`). The frontend uses Astro and requires Node/pnpm to run the local server. The two talk via standard HTTP requests.

### 4. Evaluation and Limitations
**Q: Can you explain how you identify an IR "Failure Case"?**
*A:* A failure case in our system usually occurs when a query requires knowledge from multiple, completely disjointed documents. The BM25 score might spike for one document due to keyword density, pushing other needed documents down the ranking. If the generator does not receive *both* halves of the rule in the Top-K passages, it fails to answer fully.

**Q: If you had more time, how would you improve your system?**
*A:* We could implement re-ranking (e.g., using a Cross-Encoder) on the top 50 passages returned by the hybrid model. We could also implement dynamic query routing—using an LLM to decide whether a query is keyword-heavy or semantic-heavy, and adjusting the hybrid `alpha` parameter dynamically.

### 5. Code Integrity 
*(Instructor prompt: Show me where in the code the BM25 ranking happens)*
*Preparation Action:* Group member assigned to backend should explicitly open `backend/app/engine.py` and point out the function that executes `self.bm25.get_scores(tokenized_query)`. 

*(Instructor prompt: Explain the generation config in `backend/app/services/llm.py`)*
*Preparation Action:* Point to the `generation_config` with `temperature=0.1`. Explain that low temperature means the LLM behaves highly deterministically, answering logically rather than creatively, which is vital for law.