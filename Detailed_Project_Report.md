# Comprehensive Project Documentation: Regulatory Information Retrieval and Answer Generation (RIRAG)

## Part 1: Theoretical Background — The RIRAG-2025 Shared Task
The foundational framework of this project is derived heavily from the research paper **"Shared Task RIRAG-2025: Regulatory Information Retrieval and Answer Generation"** (Gokhan et al., COLING 2025, RegNLP Workshop).

### 1.1 Introduction to the Problem Domain
Regulatory compliance is traditionally a highly complex and labor-intensive procedure. Organizations face expanding rulebooks, statutes, and obligations, often requiring meticulous manual parsing of legal documents to avoid penalties. The RIRAG-2025 shared task was established to automate these critical workflows using advanced Natural Language Processing (NLP). The task focuses on two distinct but interrelated pipelines:
1. **Subtask 1 (Information Retrieval):** Isolating and retrieving the most contextually relevant passages from a dense corpus of regulatory text.
2. **Subtask 2 (Answer Generation):** Generating precise, accurate, and highly controlled answers using exclusively the retrieved passages to guarantee compliance integrity.

### 1.2 The ObliQA Dataset
The centerpiece of the RIRAG challenge is the **ObliQA dataset**, created from Abu Dhabi Global Market (ADGM) regulations. 
- It comprises **27,869 annotated questions** mapped against 40 standard, hierarchical regulatory documents.
- Includes **Single-Passage Questions** (requiring only one specific rule to solve) and **Multi-Passage Questions** (where answers stretch across multiple rules or documents).

### 1.3 Baseline and Team Methodologies
Nineteen teams participated, exploring the intersection of sparse (lexical) search, dense (semantic) embeddings, and Large Language Models (LLMs).
- **Subtask 1 (IR) Approaches:** Most teams heavily relied on `BM25` for lexical matching, often paired with dense embeddings (such as `DRAGON+`, `ColBERTv2`, `bge-m3`, or `e5`). Fusion techniques like Reciprocal Rank Fusion or weighted hybrid systems were dominant. The top team (*Indic aiDias*) achieved a Recall@10 (R@10) of 0.787 and Mean Average Precision@10 (M@10) of 0.663.
- **Subtask 2 (Answer Generation) Approaches:** Given the strictness of legal data, prompt-engineering generative models (GPT-4, Gemini, LLaMA-3) was the primary strategy. Teams strictly limited hallucinations by feeding the retrieved context to LLMs. The leading generation score (RePASs = 0.973) produced highly stable, entailment-safe responses.

### 1.4 Lessons Learned from the Paper
The researchers identified critical insights from the competition:
1. **Underutilization of Hierarchical Data:** Most teams treated the dataset as flat chunks of text, ignoring the innate hierarchical relationships and cross-references native to the 40 raw regulatory rulebooks.
2. **Evaluation Metric Limits:** The RePASs (Regulatory Passage Answer Stability Score) metric struggled slightly to distinguish between fluent synthesis and pure verbatim parroting of the source text. Future challenges will prioritize conversational fluency and cross-document reasoning.

---

## Part 2: Detailed System Implementation

Inspired by the findings of the RIRAG-2025 shared task, our project is a fully functional, end-to-end framework built to execute both Subtasks natively. It combines an optimized hybrid retrieval algorithm with an LLM synthesizer powered by Google Gemini, wrapped in a scalable FastApi backend and a modern Astro frontend.

### 2.1 The Data & Preprocessing Layer
In `/data/`, we possess the standardized ADGM rulebook files (e.g., `AML_VER09.211223.txt`, `COBS_VER15.150823.txt`).
To optimize for Subtask 1, the `notebooks/training.ipynb` establishes the data pipeline:
- **Chunking:** The standard rulebooks are split along conceptual boundaries (defined by `passage_id` markers like `APP5.A5.7.3`). We retain 13,012 distinct passages across the 40 documents.
- **Semantic Embeddings Precomputation:** To prevent bottlenecking the server at runtime, we pass all 13,012 chunks through the `intfloat/multilingual-e5-base` Sentence Transformer. These are cached statically into `/data/processed/document_embeddings.npy`, with metadata residing in `processed_corpus.csv`.

### 2.2 The Backend (FastAPI Architecture)
Located in `/backend/`, the application logic mirrors the best methodologies (Hybrid Retrieval) discussed in the paper.

#### A. Retrieval Engine (`app/engine.py`)
This unit represents **Subtask 1**. To capture both exact legal terminology (Lexical) and conceptual intent (Semantic), we deploy a **Hybrid Retrieval** system:
- **Lexical Search:** Implemented using `BM25` (via the `rank-bm25` package). The engine tokenizes the user's query and scores it against all passages.
- **Semantic Search:** We dynamically embed the incoming user query using the same `multilingual-e5-base` model. We then leverage Scikit-Learn's `cosine_similarity` to compare the query vector against our cached `document_embeddings.npy`.
- **Score Fusion (Normalization):** Because BM25 produces unbounded scores (e.g., 15.6) and Cosine Similarity outputs bounds between 0 and 1, we apply a **Min-Max Scaler**. The two normalized scores are combined using a tunable `alpha` parameter: 
  `Final_Score = (alpha * Semantic_Score) + ((1 - alpha) * BM25_Score)`
- By default, the top-K (usually 5) highest-scoring passages are extracted and formatted as the contextual payload.

#### B. Generative Synthesizer (`app/services/llm.py`)
This represents **Subtask 2**. Instead of raw search results, business users need synthesized answers.
- We utilize Google's **Gemini 1.5 Flash** (via `google-generativeai`).
- The system prompt defines strict guardrails identical to the parameters monitored by the RePASs metric. The LLM is instructed: *"Your task is to synthesize a comprehensive, highly accurate answer. You MUST base your answer EXCLUSIVELY on the provided Context Passages. DO NOT introduce any outside information or hallucinate rules."*
- We enforce a `temperature=0.1` to ensure clinical, deterministic, factual responses.

#### C. API Routing Layer (`app/routers/`)
- **`/api/generate`:** The main pipeline endpoint. It takes the query, runs `engine.search()`, fetches Top K documents, and passes them to the Gemini generator to stream the fully formulated regulatory response back to the client. It also attaches the `passage_id` to standard outputs so sources can be verified.
- **`/api/search`:** An endpoint restricted purely to Subtask 1, meant for returning ranked passage lists without the LLM overhead.
- **`/api/document`:** Resolves a `document_id` (e.g., ID 1 maps to `AML_VER09.211223.txt` via `app/config.py`) to fetch the full raw text file for UI highlighting.

### 2.3 The Frontend (Astro Native Server)
To make the system practical, we implemented a Zero-External-Dependency UI in `/frontend/`.
- **Astro SSG/SSR Platform:** Located in `src/pages/index.astro`. The UI provides an instantaneous overview of ADGM regulations.
- **Components:**
  - A hero section featuring the central query input capability.
  - A dynamic dropdown to swap retrieval logic (Hybrid vs. Pure BM25, allowing users to benchmark the algorithms visually just like the COLING participants).
- **Document Pane Viewer:** To address the "Hierarchical isolation" issue diagnosed in the RIRAG paper, our UI implements a raw document reader. By clicking a cited source passage from the generated answer, the front end fetches the full standard `.txt` file, renders it, and produces a minimap structure. This guarantees complete transparency between the AI's generated summarized text and the actual statutory rulebook context, solving a massive usability hurdle in production RegTech tools.

### 2.4 Aligning Project Performance with Paper Takeaways
As outlined in the paper, purely generative systems (LLMs alone) fail heavily in regulatory domains. Our system perfectly mirrors the overarching lessons of the RIRAG-2025 shared task:
- **Lexical/Semantic Harmony:** Our backend `alpha` blending mimics the fusion algorithms of top leaderboard teams (like *Ocean's Eleven* and *USTC-IAT-United*).
- **Post-hoc Generation Constraints:** We address Subtask 2 by anchoring Gemini rigorously to the chunked passages retrieved by our PyTorch vectors.
- **Actionable Tooling:** Rather than just outputting CSVs of scores, the resulting project is a fully-fledged enterprise-grade scaffold that compliance officers can immediately query.


## Training Notebook
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
import pandas as pd
import numpy as np
# 40 documents as 1.json, 2.json till 40.json present in directory
RegulatoryDocumentsPath = "/content/drive/MyDrive/University/Regulatory-IR/StructuredRegulatoryDocuments/"
Loading raw JSON documents and extracting native IDs...
Successfully loaded 13012 passages with evaluation-ready IDs!
passage_id document_id text
0 1. 1 INTRODUCTION
1 1.1 1 Jurisdiction
2 1.1.1.(1) 1 The AML Rulebook is made in recognition of the...
3 1.1.1.(2) 1 Nothing in the AML Rulebook affects the operat...
4 1.2 1 Application
import os
import json
import pandas as pd
all_passages = []
print("Loading raw JSON documents and extracting native IDs...")
# Loop through 1.json to 40.json
for i in range(1, 41):
file_path = os.path.join(RegulatoryDocumentsPath, f"{i}.json")
# Check if file exists to prevent throwing an error if one is missing
if os.path.exists(file_path):
with open(file_path, 'r', encoding='utf-8') as f:
doc_data = json.load(f)
# Loop through each passage dictionary in the specific document
for item in doc_data:
# We filter out empty passages (like passage 1.1.1 in your example)
# to avoid indexing blank text into BM25 or the Dense model
passage_text = item.get("Passage", "")
if passage_text and str(passage_text).strip():
all_passages.append({
# Map the complex native ID to 'passage_id' for pytrec_eval
'passage_id': str(item.get("PassageID")),
'document_id': item.get("DocumentID"),
'text': passage_text
})
else:
print(f"Warning: {file_path} not found.")
# Create the new DataFrame, effectively replacing the old passages.csv logic
df = pd.DataFrame(all_passages)
print(f"Successfully loaded {len(df)} passages with evaluation-ready IDs!")
df.head()
df.to_csv('/content/drive/MyDrive/University/Regulatory-IR/processed_corpus.csv', index=False)
df.columns
Index(['passage_id', 'document_id', 'text'], dtype='object')
keyboard_arrow_down Model A : Baseline BM25
!pip install rank_bm25
Requirement already satisfied: rank_bm25 in /usr/local/lib/python3.12/dist-packages (0.2.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from rank_bm25) (2.0.2)
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data] Package punkt is already up-to-date!
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data] Package punkt_tab is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
def tokenize_nltk(text):
if not isinstance(text, str):
return []
text = text.lower()
tokens = word_tokenize(text)
clean_tokens = [
word for word in tokens if word.isalnum() and word not in stop_words
]
return clean_tokens
print("Starting tokenization of passages")
tokenized_corpus = df['text'].apply(tokenize_nltk).tolist()
Starting tokenization of passages
print("Building BM25 Index")
bm25 = BM25Okapi(tokenized_corpus)
print(f"Successfully built index of {len(tokenized_corpus)} passages")
Building BM25 Index
Successfully built index of 13012 passages
# defining our retrieval function
def search_bm25(query, k=10):
tokenized_query = tokenize_nltk(query)
top_n_passages = bm25.get_top_n(tokenized_query, df['text'].tolist(), n=k)
return top_n_passages
# basic testing our system
test_query = "What are the rules regarding anti-money laundering?"
results = search_bm25(test_query, k=3)
print("Top 3 Retrieved Passages:\n")
for i, passage in enumerate(results, 1):
print(f"Result {i}:\n{passage}\n")
print("-" * 50)
Top 3 Retrieved Passages:
Result 1:
A Relevant Person must ensure that it does not prejudice an Employee who discloses any information regarding money laundering to the Regulator or to any other relev
--------------------------------------------------
Result 2:
Relevant Persons should comply with guidance issued by the EOCN regarding reporting suspicious activity and Transactions relating to money laundering, terrorist fin
--------------------------------------------------
Result 3:
The GEN rules contain Rules and Guidance regarding corporate governance requirements for Authorised Persons, including the responsibilities of an Authorised Person
--------------------------------------------------
keyboard_arrow_down Model B : Multilingual E5 Base
from sentence_transformers import SentenceTransformer
import os
Loading multilingual-e5-base model
XLMRobertaModel LOAD REPORT from: intfloat/multilingual-e5-base
Key | Status | |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED | |
Notes:
- UNEXPECTED :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%  199/199 [00:00<00:00, 377.97it/s, Materializing param=pooler.dense.weight]
print("Loading multilingual-e5-base model")
model = SentenceTransformer('intfloat/multilingual-e5-base')
# prepend passage to every doc in our corpus
formatted_passages = ["passage: " + text for text in df['text'].tolist()]
Batches: 100%  407/407 [01:20<00:00, 40.20it/s]
# generating encoding here
document_embeddings = model.encode(formatted_passages, show_progress_bar=True)
# saving to our drive so we don't recompute again and again
save_path = '/content/drive/MyDrive/University/Regulatory-IR/document_embeddings.npy'
np.save(save_path, document_embeddings)
print(f"Successfully saved {len(document_embeddings)} embeddings to {save_path}!")
Successfully saved 13012 embeddings to /content/drive/MyDrive/University/Regulatory-IR/document_embeddings.npy!
from sklearn.metrics.pairwise import cosine_similarity
def search_dense(query, k=10):
# model requires query to have prefix "query: "
formatted_query = "query: " + query
# make the query a vector
query_embedding = model.encode([formatted_query])
# compute cosine similarity between query and document vectors
similarities = cosine_similarity(query_embedding, document_embeddings)[0]
# get indices of the top-k highest scoring passages
# np.argsort does asceding sorting so we sort asc then reverse by slice inversion by [::-1]
top_k_indices = np.argsort(similarities)[::-1][:k]
results = []
for ix in top_k_indices:
results.append({
'passage_id': df['passage_id'].iloc[ix],
'text': df['text'].iloc[ix],
'dense_score': similarities[ix]
})
return results
# testing our query on System B
test_query = "What are the rules regarding anti-money laundering?"
print(f"Searching for: '{test_query}'\n")
dense_results = search_dense(test_query, k=3)
print("Top 3 Retrieved Passages (Dense Retrieval):\n")
for i, result in enumerate(dense_results, 1):
print(f"Result {i} (Cosine Similarity Score: {result['dense_score']:.4f}):")
print(f"Passage ID: {result['passage_id']}")
print(f"Text: {result['text']}\n")
print("-" * 60)
Searching for: 'What are the rules regarding anti-money laundering?'
Top 3 Retrieved Passages (Dense Retrieval):
Result 1 (Cosine Similarity Score: 0.8851):
Passage ID: 3.2
Text: Anti-money laundering measures. An Authorised Person must ensure that it satisfies all the requirements that it is subject to in relation to anti-money launde
------------------------------------------------------------
Result 2 (Cosine Similarity Score: 0.8810):
Passage ID: Part 2.Chapter 1.7.(6)
Text: The Regulator may make Rules in connection with the creation and implementation of anti-money laundering measures, policies and procedures, including Rules as
(a) the persons or classes of persons who shall be subject to any such measures, policies and procedures;
(b) the nature and extent of any duty, requirement, prohibition, obligation or responsibility applicable to such persons; and
(c) registration of any or all such persons with the Regulator, including the criteria that person must meet to become and remain registered by the Regulator.
------------------------------------------------------------
Result 3 (Cosine Similarity Score: 0.8635):
Passage ID: 3.1.1
Text: A reference in the AML Rulebook to "money laundering" in lower case includes terrorist financing, proliferation financing, the financing of unlawful organisat
------------------------------------------------------------
keyboard_arrow_down System C : The Hybrid Pipeline
def min_max_scale(scores):
"""Normalizes a list of scores to 0-1 ranges."""
if not scores:
return []
min_val = min(scores)
max_val = max(scores)
# if all sscores are identical
if max_val == min_val:
return [0.5]*len(scores)
return [(x-min_val)/ (max_val - min_val) for x in scores]
def search_hybrid(query, alpha=0.5, final_k=10, candidate_k=100):
# Stage 1 : High Recall (using BM25)
tokenized_query = tokenize_nltk(query)
# Get top 100 candidates
bm25_scores = bm25.get_scores(tokenized_query)
top_100_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
candidate_bm25_scores = [bm25_scores[idx] for idx in top_100_indices]
# Stage 2 : Semantic Reranking
formatted_query = "query: " + query
query_embedding = model.encode([formatted_query])
candidate_dense_scores = []
for ix in top_100_indices:
doc_vector = document_embeddings[ix].reshape(1,-1)
sim = cosine_similarity(query_embedding, doc_vector)[0][0]
candidate_dense_scores.append(sim)
# Stage 3 : Score Fusion
norm_bm25 = min_max_scale(candidate_bm25_scores)
norm_dense = min_max_scale(candidate_dense_scores)
hybrid_results = []
for ix, original_ix in enumerate(top_100_indices):
# fusion formula
final_score = (alpha*norm_dense[ix]) + ((1-alpha)*norm_bm25[ix])
hybrid_results.append({
'passage_id': df['passage_id'].iloc[original_ix],
'text': df['text'].iloc[original_ix],
'hybrid_score': final_score,
'bm25_raw': candidate_bm25_scores[ix],
'dense_raw': candidate_dense_scores[ix]
})
# sort final 100 by hybrid_score
hybrid_results = sorted(hybrid_results, key=lambda x: x['hybrid_score'], reverse=True)
return hybrid_results[:final_k]
# Test your Hybrid System
test_query = "What are the rules regarding anti-money laundering?"
print(f"Searching for: '{test_query}' with alpha=0.5\n")
hybrid_results = search_hybrid(test_query, alpha=0.5, final_k=3)
print("Top 3 Retrieved Passages (Hybrid Retrieval):\n")
for i, result in enumerate(hybrid_results, 1):
print(f"Result {i} (Hybrid Score: {result['hybrid_score']:.4f}):")
print(f"Passage ID: {result['passage_id']}")
print(f"[Raw BM25: {result['bm25_raw']:.2f} | Raw Dense: {result['dense_raw']:.4f}]")
print(f"Text: {result['text'][:200]}...\n") # Truncated for easy reading
print("-" * 60)
Searching for: 'What are the rules regarding anti-money laundering?' with alpha=0.5
Top 3 Retrieved Passages (Hybrid Retrieval):
Result 1 (Hybrid Score: 0.8225):
Passage ID: 3.2
[Raw BM25: 10.78 | Raw Dense: 0.8851]
Text: Anti-money laundering measures. An Authorised Person must ensure that it satisfies all the requirements that it is subject to in relation to anti-money launde
------------------------------------------------------------
Result 2 (Hybrid Score: 0.7897):
Passage ID: 4.8.1
[Raw BM25: 13.86 | Raw Dense: 0.8220]
Text: A Relevant Person must ensure that it does not prejudice an Employee who discloses any information regarding money laundering to the Regulator or to any other
------------------------------------------------------------
Result 3 (Hybrid Score: 0.6595):
Passage ID: 14.3.5.Guidance.2.
[Raw BM25: 12.05 | Raw Dense: 0.8141]
Text: Relevant Persons should comply with guidance issued by the EOCN regarding reporting suspicious activity and Transactions relating to money laundering, terrori
------------------------------------------------------------
!pip install pytrec_eval
Requirement already satisfied: pytrec_eval in /usr/local/lib/python3.12/dist-packages (0.5)
QuestionID Question Passages Group
0 777e7a14-fea3-4c37-a0e6-9ffb50024d5c Can the ADGM provide clarity on the level of d... [{'DocumentID': 1, 'PassageID': '14.2.3.Guidan... 2
1 0eb99ea8-3810-492c-9986-7739006b5708 Are there any exceptions or specific circumsta... [{'DocumentID': 19, 'PassageID': '100)', 'Pass... 2
2 d34e3516-f053-4652-a0ac-ede703144b9a What type of procedures must a Third Party Pro... [{'DocumentID': 3, 'PassageID': '20.14.1.(2)',... 1
3 6d876d32-7557-4149-875e-8c66f13f3485 Are there any exceptions or specific circumsta... [{'DocumentID': 13, 'PassageID': '4.15.16', 'P... 3
4 2efd28f4-8677-4f05-82cd-d9989fb72409 What specific areas of inventory and delivery ... [{'DocumentID': 34, 'PassageID': '35)', 'Passa... 3
test_df = pd.read_json("/content/drive/MyDrive/University/Regulatory-IR/ObliQA_test.json")
test_df.head()
"""
Sample:
{
"QuestionID": "777e7a14-fea3-4c37-a0e6-9ffb50024d5c",
"Question": "Can the ADGM provide clarity on the level of detail and documentation that should accompany a report of suspicious activity to ensu
"Passages": [
{
"DocumentID": 1,
"PassageID": "14.2.3.Guidance.10.",
"Passage": "Relevant Persons should comply with guidance issued by the EOCN with regard to identifying and reporting suspicious activity
}
],
"Group": 2
},
"""
test_df.columns
Index(['QuestionID', 'Question', 'Passages', 'Group'], dtype='object')
import pytrec_eval
import numpy as np
# 1. Prepare the Ground Truth (qrels) from your test_df
qrels = {}
for index, row in test_df.iterrows():
qid = row['QuestionID']
qrels[qid] = {}
# Iterate through the list of correct passages for this specific question
for passage in row['Passages']:
correct_pid = passage['PassageID']
# Assign a relevance score of 1 (meaning it is a correct document)
qrels[qid][correct_pid] = 1
# 2. Helper to format BM25 output for pytrec_eval
def get_bm25_predictions(query, k=10):
tokenized_query = tokenize_nltk(query)
scores = bm25.get_scores(tokenized_query)
top_k_indices = np.argsort(scores)[::-1][:k]
# Return a dictionary of {passage_id: score}
return {df['passage_id'].iloc[idx]: float(scores[idx]) for idx in top_k_indices}
# 3. Prepare the Runs (Predictions) for each system
run_system_a = {}
run_system_b = {}
run_system_c = {}
print(f"Evaluating {len(test_df)} test questions... (This might take a moment)")
for index, row in test_df.iterrows():
qid = row['QuestionID']
query_text = row['Question']
# --- System A (BM25) ---
run_system_a[qid] = get_bm25_predictions(query_text, k=10)
# --- System B (Dense) ---
results_b = search_dense(query_text, k=10)
run_system_b[qid] = {res['passage_id']: float(res['dense_score']) for res in results_b}
# --- System C (Hybrid) ---
results_c = search_hybrid(query_text, alpha=0.5, final_k=10)
run_system_c[qid] = {res['passage_id']: float(res['hybrid_score']) for res in results_c}
# 4. Define the metrics we want to calculate
metrics = {'recall_10', 'map_cut_10', 'ndcg_cut_10'}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
# 5. Calculate and print the results
def print_results(system_name, run_data):
results = evaluator.evaluate(run_data)
# Average the scores across all queries
avg_recall = np.mean([query_metrics['recall_10'] for query_metrics in results.values()])
avg_map = np.mean([query_metrics['map_cut_10'] for query_metrics in results.values()])
avg_ndcg = np.mean([query_metrics['ndcg_cut_10'] for query_metrics in results.values()])
print(f"--- {system_name} Performance ---")
print(f"Recall@10: {avg_recall:.4f}")
print(f"MAP@10: {avg_map:.4f}")
print(f"NDCG@10: {avg_ndcg:.4f}\n")
# Run the final evaluation printout!
print_results("System A (BM25 Baseline)", run_system_a)
print_results("System B (Dense Retrieval)", run_system_b)
print_results("System C (Hybrid Pipeline)", run_system_c)
Evaluating 2786 test questions... (This might take a moment)
--- System A (BM25 Baseline) Performance ---
Recall@10: 0.7354
MAP@10: 0.5759
NDCG@10: 0.6312
--- System B (Dense Retrieval) Performance ---
Recall@10: 0.7169
MAP@10: 0.5637
NDCG@10: 0.6151
--- System C (Hybrid Pipeline) Performance ---
Recall@10: 0.7762
MAP@10: 0.6282
NDCG@10: 0.6816
def search_hybrid_rrf(query, k_constant=60, final_k=10, candidate_k=100):
# ---------------------------------------------------------
# 1. Get Top Candidates & Ranks from BM25
# ---------------------------------------------------------
tokenized_query = tokenize_nltk(query)
bm25_scores = bm25.get_scores(tokenized_query)
bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
# ---------------------------------------------------------
# 2. Get Top Candidates & Ranks from Dense
# ---------------------------------------------------------
formatted_query = "query: " + query
query_embedding = model.encode([formatted_query])
# Compute similarity against the whole corpus to get true ranks
similarities = cosine_similarity(query_embedding, document_embeddings)[0]
dense_top_indices = np.argsort(similarities)[::-1][:candidate_k]
# ---------------------------------------------------------
# 3. Apply Reciprocal Rank Fusion
# ---------------------------------------------------------
rrf_scores = {}
# Process BM25 ranks (enumerate starts at 1 for Rank 1)
for rank, idx in enumerate(bm25_top_indices, 1):
if idx not in rrf_scores:
rrf_scores[idx] = 0.0
rrf_scores[idx] += 1.0 / (k_constant + rank)
# Process Dense ranks
for rank, idx in enumerate(dense_top_indices, 1):
if idx not in rrf_scores:
rrf_scores[idx] = 0.0
rrf_scores[idx] += 1.0 / (k_constant + rank)
# ---------------------------------------------------------
# 4. Sort & Format Results
# ---------------------------------------------------------
# Sort the dictionary keys (indices) by their accumulated RRF score descending
sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:final_k]
hybrid_results = []
for idx in sorted_indices:
hybrid_results.append({
'passage_id': df['passage_id'].iloc[idx],
'text': df['text'].iloc[idx],
'hybrid_score': rrf_scores[idx]
})
return hybrid_results
run_system_rrf = {}
print(f"Evaluating System D (RRF Hybrid) over {len(test_df)} test questions...")
for index, row in test_df.iterrows():
qid = row['QuestionID']
query_text = row['Question']
# Run the new RRF Hybrid
results_rrf = search_hybrid_rrf(query_text, k_constant=60, final_k=10)
# Format the predictions for pytrec_eval
run_system_rrf[qid] = {res['passage_id']: float(res['hybrid_score']) for res in results_rrf}
# Print the final metrics using the evaluator we defined in the previous phase
print_results("System D (RRF Hybrid Pipeline)", run_system_rrf)
Evaluating System D (RRF Hybrid) over 2786 test questions...
--- System D (RRF Hybrid Pipeline) Performance ---
Recall@10: 0.7697
MAP@10: 0.6137
NDCG@10: 0.6679