import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from app.config import PROCESSED_CORPUS_PATH, EMBEDDINGS_PATH, MODEL_NAME

class RetrievalEngine:
    def __init__(self):
        self.df = None
        self.document_embeddings = None
        self.model = None
        self.bm25 = None
        self.stop_words = None

    def load_resources(self):
        print("Loading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

        print("Loading dataset and embeddings...")
        self.df = pd.read_csv(PROCESSED_CORPUS_PATH)
        self.document_embeddings = np.load(EMBEDDINGS_PATH)

        print(f"Loading dense model ({MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Building BM25 Index...")
        tokenized_corpus = self.df['text'].apply(self.tokenize_nltk).tolist()
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ Retrieval Engine Fully Loaded and Ready!")

    def tokenize_nltk(self, text):
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if w.isalnum() and w not in self.stop_words]

    def _min_max_scale(self, scores):
        if not scores: return []
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val: return [0.5] * len(scores)
        return [(x - min_val) / (max_val - min_val) for x in scores]

    def search(self, query: str, system: str = "hybrid", k: int = 10):
        if system == "bm25":
            return self._search_bm25(query, k)
        elif system == "dense":
            return self._search_dense(query, k)
        elif system == "hybrid":
            return self._search_hybrid(query, k)
        else:
            raise ValueError("Invalid system type selected.")

    def _format_results(self, indices, scores):
        results = []
        for i, idx in enumerate(indices):
            results.append({
                "passage_id": str(self.df['passage_id'].iloc[idx]),
                "document_id": int(self.df['document_id'].iloc[idx]),
                "document_name": str(self.df.get('document_name', self.df['document_id']).iloc[idx]),
                "text": str(self.df['text'].iloc[idx]),
                "score": float(scores[i])
            })
        return results

    def _search_bm25(self, query, k):
        tokenized_query = self.tokenize_nltk(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = [scores[i] for i in top_indices]
        return self._format_results(top_indices, top_scores)

    def _search_dense(self, query, k):
        formatted_query = "query: " + query
        # convert_to_numpy prevents Pylance/Sklearn Tensor errors!
        query_embedding = self.model.encode([formatted_query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = [similarities[i] for i in top_indices]
        return self._format_results(top_indices, top_scores)

    def _search_hybrid(self, query, k, alpha=0.5):
        tokenized_query = self.tokenize_nltk(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_100_indices = np.argsort(bm25_scores)[::-1][:100]
        candidate_bm25_scores = [bm25_scores[idx] for idx in top_100_indices]

        formatted_query = "query: " + query
        query_embedding = self.model.encode([formatted_query], convert_to_numpy=True)
        
        candidate_dense_scores = []
        for idx in top_100_indices:
            doc_vector = self.document_embeddings[idx].reshape(1, -1)
            sim = cosine_similarity(query_embedding, doc_vector)[0][0]
            candidate_dense_scores.append(float(sim))
            
        norm_bm25 = self._min_max_scale(candidate_bm25_scores)
        norm_dense = self._min_max_scale(candidate_dense_scores)
        
        hybrid_results = []
        for i, original_idx in enumerate(top_100_indices):
            final_score = (alpha * norm_dense[i]) + ((1 - alpha) * norm_bm25[i])
            hybrid_results.append({
                "passage_id": str(self.df['passage_id'].iloc[original_idx]),
                "document_id": int(self.df['document_id'].iloc[original_idx]),
                "document_name": str(self.df.get('document_name', self.df['document_id']).iloc[original_idx]),
                "text": str(self.df['text'].iloc[original_idx]),
                "score": final_score
            })
            
        return sorted(hybrid_results, key=lambda x: x['score'], reverse=True)[:k]