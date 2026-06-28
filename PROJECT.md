# Regulatory IR: RIRAG 2025 Shared Task

## Short Description
A Regulatory Information Retrieval and Answer Generation (RIRAG) system for ADGM regulations. It retrieves relevant regulatory passages with a hybrid search pipeline and generates grounded, concise answers using an LLM constrained to the retrieved context.

## Main Techniques and Tools Used
- Hybrid retrieval (BM25 + dense embeddings) with score normalization
- SentenceTransformers for dense semantic embeddings
- BM25 lexical retrieval (rank-bm25)
- NLTK tokenization and stopword filtering
- Cosine similarity via scikit-learn
- FastAPI backend API with modular routers
- Gemini API for answer generation (context-grounded RAG)
- Astro frontend for the web UI

## Project Selected From
RegNLP 2025 Workshop at COLING 2025, RIRAG shared task:
- https://aclanthology.org/2025.regnlp-1.1/

## Dataset Source
ObliQA Dataset (ADGM regulatory compliance QA dataset):
- https://github.com/RegNLP/ObliQADataset
