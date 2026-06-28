# Regulatory IR : RIRAG 2025 Shared Task

The task was organized in time for COLING 2025
as part of the RegNLP 2025 workshop.

The Regulatory Information Retrieval and
Answer Generation (RIRAG) focuses on automating two core processes: retrieving relevant regulatory information and generating concise, accurate
answers to compliance-related questions. By combining information retrieval and answer generation,
RIRAG provides a framework to streamline compliance workflows and enhance organizational efficiency.

## Dataset
The shared task leverages the ObliQA dataset 1,
a regulatory compliance-focused dataset derived
from Abu Dhabi Global Market (ADGM) regulations. ObliQA comprises 27,869 questions, each
annotated with corresponding passages, making it
a robust resource for developing and benchmarking
RIRAG systems. The dataset poses unique challenges, including:
Single-Passage Questions: Questions that require retrieving and analyzing a single passage.
Multi-Passage Questions: Questions necessitating the integration of multiple passages for a complete answer.

## Evaluation
To evaluate system performance, different metrics
are applied to the two subtasks. For Subtask 1
(Information Retrieval), Recall at 10 (R@10) and
Mean Average Precision at 10 (M@10) are used
to assess the system’s ability to retrieve relevant
passages effectively.

For Subtask 2 (Answer Generation), the Regulatory Passage Answer Stability
Score (RePASs)2 measures the quality of generated
answers based on their entailment with source passages, avoidance of contradictions, and coverage
of obligations.

---

## Installation & Execution Instructions (Term Project)

Follow these step-by-step instructions to run the application locally for the project demonstration.

### Prerequisites
- Python 3.10+
- Node.js (v18+) and standard `pnpm` (or `npm`)
- A `.env` file created in the `backend/` directory with a valid Google Gemini API Key.

### 1. Setup the Backend (FastAPI + Retrieval Engine)
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `backend/` root and add your API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. Start the backend Server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *Note: Ensure your `processed` dataset files (like `document_embeddings.npy` and `processed_corpus.csv`) are located inside the `data/processed/` directory.*

### 2. Setup the Frontend (Astro)
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the frontend dependencies:
   ```bash
   pnpm install
   ```
3. Start the Astro development server:
   ```bash
   pnpm dev
   ```
4. The web interface will be accessible at `http://localhost:4321`.

### 3. Running a Query (Demo)
- Open `http://localhost:4321` in your browser.
- Type a regulatory query such as: *"What are the rules regarding anti-money laundering?"*
- Select "HYBRID" search and submit.
- The web app will query the backend, rank the passages, generate the RAG answer, and display the relevant documents highlighted in the UI.