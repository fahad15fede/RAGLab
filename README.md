
# VIPER AI — Conversational RAG Chatbot & Retrieval Benchmarking Lab

**A documentation-grounded AI programming assistant for the custom VIPER language, with an integrated retrieval benchmarking framework.**

Built by **Muhammad Fahad Pervez**

[GitHub Profile](https://github.com/fahad15fede)

---

## Project Overview

VIPER AI is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about **VIPER**, a custom programming language with its own syntax, keywords, data types, grammar, and lexical rules.

The project began as a documentation-grounded chatbot and evolved into **RAGLab**, an experimental framework for evaluating different information retrieval techniques.

Rather than relying on a single retrieval method, I implemented and compared:

- BM25 (Keyword-Based Retrieval)
- Semantic Search (Vector Similarity)
- Hybrid Retrieval (BM25 + Semantic using Reciprocal Rank Fusion)
- Maximal Marginal Relevance (MMR)

The objective was to understand how retrieval methods differ in latency, ranking behavior, diversity, and document selection.

**Project Status:** V1 — Functional chatbot with an experimental retrieval benchmarking suite.

> The chatbot currently uses semantic retrieval with exact VIPER keyword matching. The RAGLab retrieval strategies are implemented and benchmarked separately. A production switch to one of these experimental strategies has not yet been established.

---

## Why I Built This Project

Large Language Models are trained on widely available programming languages such as Python, Java, and C++.

However, VIPER is a custom language whose syntax is not reliably known to general-purpose LLMs.

For example:

```viper
HOLD
    fang age = 20;

    strike (age >= 18) {
        hissout("Adult");
    } slither {
        hissout("Minor");
    }
RELEASE
```

Without access to VIPER documentation, an LLM might generate incorrect syntax or hallucinate language features.

I wanted to build a system that:

1. Understands questions about VIPER.
2. Retrieves relevant information from official project documentation.
3. Generates grounded answers using an LLM.
4. Supports follow-up questions and conversational context.
5. Provides references to the retrieved documentation.
6. Allows experimentation with different retrieval strategies.

---

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| LLM | Google Gemini |
| Embeddings | gemini-embedding-001 |
| Vector Database | Chroma Cloud |
| RAG Framework | LangChain |
| Workflow Orchestration | LangGraph |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Diversity-Aware Retrieval | MMR |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Testing | Python unittest |
| Version Control | Git and GitHub |

---

## System Architecture

```text
                     USER QUESTION
                           |
                           v
                    FastAPI Backend
                           |
                           v
                    LangGraph Router
                           |
              +------------+------------+
              |                         |
              v                         v
        VIPER Question            Other Request
              |                         |
              v                         v
      Question Rewriting          Direct Response
              |
              v
       Retrieval Pipeline
              |
      +-------+--------+
      |                |
      v                v
Semantic Search   Exact Keyword Matching
      |                |
      +-------+--------+
              |
              v
     Deduplicate Results
              |
              v
      Retrieved Context
              |
              v
          Gemini LLM
              |
              v
     Grounded AI Response


          EXPERIMENTAL RAGLAB

                Query
                  |
    +-------------+-------------+
    |             |             |
    v             v             v
   BM25        Semantic       Hybrid
    |             |             |
    |             |             v
    |             |            RRF
    |             |
    |             +----------> MMR
    |
    +-------------+-------------+
                  |
                  v
          Benchmark Engine
                  |
                  v
       Latency / Overlap Analysis
                  |
                  v
            CSV + Charts
```

---

# Development Journey

## Phase 1 — Building the VIPER Knowledge Base

The first step was creating a documentation-based knowledge source.

The documents contain information about:

- VIPER data types
- Variable declarations
- Initialization rules
- Operators and expressions
- Conditional statements
- Loops and functions
- Input/output syntax
- Lexical errors
- Grammar rules and code examples

### Knowledge Sources

```text
VIPER_Programming_Language.pdf
VIPER_CFG_Code_Samples.pdf
VIPER_Lexical_Errors_Detailed_RAG_Guide.pdf
```

These documents serve as the chatbot's primary knowledge base.

---

## Phase 2 — PDF Processing and Chunking

I implemented a document ingestion pipeline using LangChain.

### Workflow

```text
PDF Documents
     |
     v
PDF Loader
     |
     v
Text Cleaning
     |
     v
RecursiveCharacterTextSplitter
     |
     v
Text Chunks
     |
     v
Gemini Embeddings
     |
     v
Chroma Cloud
```

### Chunking Configuration

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

Chunk overlap helps preserve context when information spans chunk boundaries.

### Embedding Model

```python
GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)
```

Each chunk is transformed into a 768-dimensional embedding vector.

The experimental collection contained **78 indexed chunks** during the reported benchmark runs.

### Metadata Preservation

Each indexed chunk preserves useful metadata, including:

- Source document
- Page number
- Original chunk content
- Stable chunk identifier

Stable IDs are generated using document information and content hashing.

This helps identify chunks consistently across retrieval experiments.

---

## Phase 3 — Building the AI Chatbot

After indexing the documents, I developed the chatbot using LangGraph and Gemini.

### LangGraph Workflow

The chatbot includes:

**1. Query Routing**

Classifies questions into:

- `rag`
- `greeting`
- `out_of_scope`

**2. Question Rewriting**

Uses conversation history to rewrite context-dependent follow-up questions.

**3. Retrieval**

Retrieves relevant documentation chunks from Chroma Cloud.

**4. Context Construction**

Combines retrieved chunks with source references.

**5. Answer Generation**

Passes the retrieved context to Gemini to generate documentation-grounded responses.

### Conversational Memory

LangGraph's `InMemorySaver` is used to maintain conversation state during the application process.

This supports follow-up questions but does not persist conversations across server restarts.

---

## Phase 4 — FastAPI Integration

The chatbot was exposed through a FastAPI backend.

### Main Endpoints

```http
GET /health
POST /chat
```

Example request:

```json
{
  "message": "How do I declare a fang variable?",
  "session_id": "demo-session"
}
```

The API processes the query through LangGraph and returns an AI-generated response with supporting source references.

This separates the application interface from the retrieval and generation logic.

---

## Phase 5 — Improving Retrieval

During development, I realized that retrieval is one of the most important components of a RAG system.

VIPER contains short, domain-specific keywords such as:

```text
fang
venom
coil
strike
shift
slither
```

Pure semantic retrieval may not always prioritize exact syntax matches.

To address this, the chatbot combines:

- Semantic similarity search
- Exact VIPER keyword matching
- Query expansion for selected lexical-validation questions
- Result deduplication

This is the retrieval approach currently integrated into the chatbot.

I then built RAGLab to investigate additional retrieval strategies independently.

---

# RAGLab — Retrieval Benchmarking Framework

## Phase 6 — BM25 Retrieval

BM25 is a keyword-based ranking algorithm.

It considers:

- Term frequency
- Inverse document frequency
- Document length normalization

Example query:

```text
How do I declare a fang variable?
```

BM25 ranks chunks containing relevant terms such as `fang`, `variable`, and `declaration`.

### Implementation

Used the `rank_bm25` library to build a local index from the Chroma document collection.

**Advantage:** Extremely low measured local query latency.

**Limitation:** Keyword matching may miss relevant documents expressed using different terminology.

---

## Phase 7 — Semantic Retrieval

Semantic retrieval uses vector embeddings to find documents with similar meanings.

### Workflow

```text
User Query
    |
    v
Gemini Embedding
    |
    v
768-Dimensional Vector
    |
    v
Chroma Cloud Search
    |
    v
Top-K Relevant Chunks
```

Semantic retrieval can identify conceptually related content even when the wording differs.

However, remote embedding and vector search introduce additional latency.

---

## Phase 8 — Hybrid Retrieval Using RRF

Hybrid retrieval combines keyword-based and semantic retrieval.

I implemented **Reciprocal Rank Fusion (RRF)** to merge their rankings.

### Architecture

```text
             User Query
                 |
       +---------+---------+
       |                   |
       v                   v
      BM25              Semantic
       |                   |
       v                   v
  Ranked Chunks       Ranked Chunks
       |                   |
       +---------+---------+
                 |
                 v
       Reciprocal Rank Fusion
                 |
                 v
          Final Ranking
```

### RRF Formula

```text
RRF(d) = Σ [1 / (k + rank(d))]
```

Where:

- `d` is a document.
- `k` is a smoothing constant, commonly 60.
- `rank(d)` is the document's position in a ranked list.

Documents appearing in multiple retrieval lists can benefit from contributions from both lists.

I also extracted RRF ranking logic into reusable functions for independent unit testing.

---

## Phase 9 — Maximal Marginal Relevance (MMR)

MMR balances relevance and diversity.

Standard semantic retrieval may return several chunks containing nearly identical information.

MMR attempts to reduce redundancy by selecting documents that are relevant to the query but less similar to already selected documents.

### Formula

```text
MMR(d) =
    λ × similarity(query, document)
    -
    (1 - λ) × max_similarity(document, selected_documents)
```

The parameter `lambda_mult` controls the trade-off.

- `λ = 1.0`: Focus on relevance.
- `λ = 0.5`: Balance relevance and diversity.
- `λ = 0.2`: Give more weight to diversity.

I implemented cosine similarity and greedy MMR selection, then refactored the retriever to reuse the tested ranking functions.

---

# Phase 10 — Benchmarking All Four Retrievers

After implementing the retrieval strategies, I created a benchmarking framework.

The second benchmark used:

- 4 evaluation questions
- 4 retrieval strategies
- 5 measured runs per question and retriever
- Warm-up calls before measurements

**Total: 80 timed retrieval operations.**

## Latency Benchmark Results

| Retriever | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| BM25 | 0.24 | 0.23 | 0.42 |
| Semantic | 665.72 | 620.39 | 881.22 |
| Hybrid RRF | 846.06 | 820.16 | 997.41 |
| MMR | 1342.10 | 1054.32 | 2384.45 |

### Key Observations

**BM25**

The fastest measured retrieval strategy because it searched an already-loaded local index.

**Semantic Retrieval**

Required embedding generation and remote Chroma Cloud search.

**Hybrid RRF**

Combined BM25 and semantic results, introducing additional retrieval and fusion work.

**MMR**

Had the highest mean latency in this experiment because of candidate retrieval and diversity-aware selection.

These results describe the tested implementation and environment, not universal algorithm performance.

---

## Retrieval Overlap Analysis

I also compared retrieval results using **Jaccard similarity**.

### Formula

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

This measures how many retrieved chunks two strategies have in common.

### Results

| Retrieval Pair | Average Jaccard Similarity |
|---|---:|
| Semantic vs MMR | 0.503 |
| BM25 vs Hybrid | 0.364 |
| Semantic vs Hybrid | 0.364 |
| BM25 vs Semantic | 0.214 |
| Hybrid vs MMR | 0.163 |
| BM25 vs MMR | 0.135 |

### Interpretation

Semantic and MMR shared the highest proportion of retrieved chunks among the compared pairs.

BM25 and MMR had the lowest overlap.

This demonstrates that retrieval strategies can produce substantially different context sets.

**Important:** Jaccard similarity measures overlap, not retrieval correctness or answer quality.

---

# Phase 11 — Unit Testing and Refactoring

I implemented reusable ranking functions and tested them independently of cloud services.

### Ranking Functions

```python
reciprocal_rank_fusion()
cosine_similarity()
mmr_select()
```

### Unit Tests

Six ranking tests passed:

```text
test_cosine_identical_vectors ... ok
test_cosine_orthogonal_vectors ... ok
test_invalid_lambda ... ok
test_mmr_diversity ... ok
test_mmr_relevance_only ... ok
test_rrf_favors_shared_chunks ... ok

Ran 6 tests

OK
```

### Hybrid Integration Testing

I also created a mocked Hybrid integration test.

The test verifies:

- BM25 and semantic ranking combination
- Correct fused document ordering
- Valid RRF scores
- Expected result count

The test passed without requiring Gemini or Chroma API calls.

### Refactoring

Originally, ranking logic was implemented directly inside the retrieval classes.

I later separated ranking calculations from database interaction.

This improved:

- Code reusability
- Testability
- Separation of concerns
- Maintainability

---

# Engineering Challenges and Lessons Learned

## 1. Retrieval Is More Than Vector Search

Semantic search is useful, but domain-specific keywords can benefit from lexical matching.

Different retrieval strategies serve different requirements.

## 2. Fastest Does Not Mean Most Accurate

BM25 was significantly faster in the benchmark.

However, latency alone does not prove that it retrieves the most relevant information.

## 3. Ground-Truth Evaluation Is Important

The evaluation dataset currently contains questions without labeled relevant chunk IDs.

Therefore, the project does not claim measured:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG

These metrics require ground-truth relevance labels.

## 4. Diversity Can Improve Context Coverage

MMR provides a way to reduce repeated information in retrieved context.

However, increasing diversity can also reduce relevance.

## 5. Cloud Services Affect Latency

Gemini embedding requests and Chroma Cloud queries introduce network-dependent delays.

Benchmarks must account for service response time, warm-up effects, and candidate-pool size.

## 6. Testable Components Make Experimentation Easier

Separating RRF and MMR ranking functions from retrieval infrastructure allowed unit testing without external APIs.

---

# Repository Structure

```text
.
├── api.py
├── viper_graph.py
├── vector_store.py
├── chroma_client.py
│
├── data/
│   ├── VIPER_Programming_Language.pdf
│   ├── VIPER_CFG_Code_Samples.pdf
│   └── VIPER_Lexical_Errors_Detailed_RAG_Guide.pdf
│
├── rag_lab/
│   ├── retrievers/
│   │   ├── base.py
│   │   ├── bm25.py
│   │   ├── semantic.py
│   │   ├── hybrid.py
│   │   ├── mmr.py
│   │   └── rankings.py
│   │
│   ├── evaluation/
│   │   └── dataset.json
│   │
│   ├── experiments/
│   │   ├── benchmark.py
│   │   ├── benchmark_v2.py
│   │   ├── compare.py
│   │   ├── test_semantic.py
│   │   ├── test_hybrid.py
│   │   └── test_mmr.py
│   │
│   └── results/
│       ├── benchmark.csv
│       ├── leaderboard.csv
│       ├── benchmark_v2_runs.csv
│       ├── benchmark_v2_summary.csv
│       ├── retrieval_overlap.csv
│       ├── latency_comparison.png
│       └── source_diversity.png
│
├── tests/
│   ├── test_rag_metrics.py
│   ├── test_rankings.py
│   └── test_hybrid_integration.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The directory tree represents the intended combined repository. Verify that all listed files are committed.

---

# Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/fahad15fede/RAGLab.git
cd RAGLab
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file:

```dotenv
GOOGLE_API_KEY=your_google_api_key
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database
CHROMA_COLLECTION=your_collection_name
GEMINI_MODEL=your_available_gemini_chat_model
```

Never commit your actual `.env` file or API credentials.

The Chroma collection must use embeddings compatible with the project's 768-dimensional Gemini vectors.

## 5. Index VIPER Documentation

Place the documentation PDFs in the `data/` directory.

Run:

```bash
python vector_store.py
```

This processes the PDFs, generates embeddings, and stores the document chunks in Chroma Cloud.

## 6. Start the FastAPI Server

```bash
uvicorn api:app --reload --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

## 7. Run Retrieval Experiments

```bash
python -m rag_lab.experiments.test_hybrid
python -m rag_lab.experiments.test_mmr
python -m rag_lab.experiments.benchmark_v2
```

## 8. Generate Benchmark Comparisons

```bash
python -m rag_lab.experiments.compare
```

## 9. Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

# Deployment

The FastAPI application can be deployed to a Python hosting service such as Render.

Example build command:

```bash
pip install -r requirements.txt
```

Example start command:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Configure all API credentials through the hosting platform's environment variables.

A public production deployment would require additional security, rate limiting, monitoring, and persistent conversation storage.

---

# Future Improvements

- Add labeled relevance datasets.
- Evaluate Precision@K, Recall@K, MRR, and nDCG.
- Compare retrieval strategies using generated-answer quality.
- Add cross-encoder or LLM-based reranking.
- Experiment with Multi-Query Retrieval.
- Add configurable retrieval strategies to the live chatbot.
- Improve automated testing and benchmarking.
- Add persistent conversation memory.
- Introduce production monitoring and observability.

---

# Final Takeaways

This project represents my progression from developing a functional RAG chatbot to investigating retrieval engineering through practical experiments.

I learned how to:

- Build a document-grounded conversational assistant.
- Process and index PDF knowledge bases.
- Use Gemini embeddings with Chroma Cloud.
- Build LangGraph-based conversational workflows.
- Implement lexical and semantic retrieval.
- Combine rankings using Reciprocal Rank Fusion.
- Implement diversity-aware retrieval using MMR.
- Benchmark retrieval performance.
- Analyze result-set overlap.
- Refactor retrieval algorithms into reusable, testable components.

The biggest takeaway was that **retrieval engineering is about balancing relevance, latency, diversity, and system complexity—not simply choosing the fastest algorithm.**

---

## Author

**Muhammad Fahad Pervez**

GitHub: [fahad15fede](https://github.com/fahad15fede)

**Project:** VIPER AI & RAGLab

**Version:** 1.0 — Experimental Release
