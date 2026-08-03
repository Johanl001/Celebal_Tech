Document Question Answering System — Retrieval-Augmented Generation (RAG)

Week 7 Assignment | Data Science Internship

Author: Dhanraj Deshmukh

Overview
This project builds a Retrieval-Augmented Generation pipeline that answers questions grounded in a custom document instead of relying on a language model's internal knowledge. A document — uploaded directly by the user, or pulled from a Hugging Face dataset — is ingested, split into overlapping chunks, converted into embeddings, and stored in a vector database. At query time, the question is embedded with the same model, the most relevant chunks are retrieved, and a language model generates an answer using only that retrieved context — which keeps the answers factually tied to the source document rather than the model's own guesses.

Note on references: an earlier draft of this README cited VivekChauhan05/RAG_Document_Question_Answering as the structural reference for this pipeline. That repo actually uses a different stack (Cohere for embeddings/generation, Pinecone as the vector store, Streamlit as the interface), which this notebook does not follow, so the citation has been removed. This implementation is a self-contained, key-free alternative built around open-source, locally runnable components instead.

Project Structure
```
Document_QA_RAG_Week7.ipynb   ← Main notebook (fully self-contained, demo document embedded)
README.md                       ← This file
```

Dataset
Source: user-uploaded PDF or TXT file (via an in-notebook upload widget), a Hugging Face dataset (e.g. `vectara/open_ragbench`), or a fallback demo document (internship program guidelines) embedded directly in the notebook if nothing is uploaded

| Property | Value |
|---|---|
| Document type | PDF, TXT, Hugging Face dataset, or the built-in demo text — all handled by the same `load_document()` function |
| Document length | Depends on the uploaded document; demo fallback is a few paragraphs |
| Chunk size | 300 characters, 50-character overlap |
| Total chunks | Derived from chunking the document (printed on run) |
| Labels | Not applicable — this is retrieval + generation, not supervised classification |

Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Setup | Install and import chunking, embedding, vector store, and generation libraries |
| 2 | Document Ingestion | `load_document()` handles PDF, TXT, raw text, or a Hugging Face dataset; an in-notebook upload widget lets the user supply a real file directly, with the demo document as fallback |
| 3 | Text Chunking | `RecursiveCharacterTextSplitter` breaks the document into overlapping chunks |
| 4 | Embedding Creation | Each chunk converted to a 384-dimensional vector using `all-MiniLM-L6-v2` |
| 5 | Vector Database | Embeddings stored in a ChromaDB collection for similarity search |
| 6 | Query Processing | User question embedded with the same embedding model as the document chunks |
| 7 | Context Retrieval | `retrieve_context()` queries ChromaDB for the top-k most relevant chunks |
| 8 | Answer Generation | Retrieved chunks and the question combined into a prompt, answered by `flan-t5-base` |
| 9 | Validation | Full pipeline run on 4 different sample questions, logged with retrieval distance and response time |
| 10 | System Metrics Report | Chunking, embedding, vector store, and generation configuration summarized in a table |
| 11 | Optimization Experiments | Chunk size comparison, hybrid keyword+vector search, and re-ranking with a real cross-encoder model |

Architecture & Design

| Artifact | Formula / Logic | Rationale |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)` | Keeps chunks small enough for targeted retrieval while overlap preserves context across chunk boundaries |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | Small enough to run on CPU, common default for RAG prototypes |
| Vector store | ChromaDB in-memory client | Handles similarity search without needing to implement nearest-neighbor search manually |
| Generation model | `google/flan-t5-base` | Runs locally without an API key, keeps the notebook fully self-contained and reproducible |
| Hybrid retrieval | `vector_weight * vector_score + (1 - vector_weight) * keyword_score` | Vector search alone can miss exact-keyword matches; combining both scores catches more relevant chunks |
| Re-ranking | Pull top 10 by vector similarity, re-score with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep top 3 | A real cross-encoder scores the question and chunk jointly rather than comparing separate embeddings, which is typically more accurate than vector similarity alone — used as a second pass since it's slower per candidate |

Components Used

| Component | Tool |
|---|---|
| Document parsing | `pypdf` (PDF), built-in file I/O (TXT), `datasets` (Hugging Face) |
| Document upload | `google.colab.files` upload widget |
| Text chunking | `langchain-text-splitters` |
| Embedding model | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | `chromadb` |
| Language model | `transformers` pipeline (`google/flan-t5-base`) |
| Re-ranking model | `sentence-transformers` `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |

Model Validation
- Retrieval checked against 4 different sample questions covering different sections of the document, not just one easy case.
- Each validation run logs the closest retrieval distance, number of chunks retrieved, and response time, tabulated in a results DataFrame.
- Chunk size compared across three settings (150 / 300 / 600 characters) with retrieval distance for the same test question, to check the effect of chunking on retrieval quality directly rather than assuming a fixed size is correct.
- Hybrid search and re-ranking results inspected manually against plain vector search to confirm they surface different (and in some cases better-matched) chunks.

Key Findings
(fill in the specific numbers after running the notebook — the cells are already built to produce each of these)

| Stage | Finding |
|---|---|
| Retrieval accuracy | Whether the top-retrieved chunk is actually the correct source for each sample question — record here |
| Chunk size effect | Which of the three tested chunk sizes gave the lowest (best) retrieval distance — record here |
| Hybrid vs pure vector search | Cases where hybrid search retrieved a more relevant chunk than vector search alone — record here |
| Generation quality | Whether flan-t5-base answers stayed grounded in context or drifted, across the 4 validation questions — record here |
| Response time | Average end-to-end response time per question — record here |
| Failure cases | Question types the pipeline struggled with (e.g. answers requiring multiple chunks) — record here |

Tech Stack

| Library | Purpose |
|---|---|
| langchain-text-splitters | Document chunking |
| sentence-transformers | Text embedding and cross-encoder re-ranking |
| chromadb | Vector storage and similarity search |
| transformers | Answer generation |
| pypdf | PDF text extraction |
| datasets | Hugging Face dataset ingestion |
| pandas | Validation log and metrics report tables |

How to Run
```
# 1. No dataset download needed by default — a demo document is embedded
#    directly in the notebook. To use a real document instead, just run
#    the upload cell in Section 2.1 and pick a PDF or TXT file when
#    prompted (this only works in Google Colab). To pull a document from
#    Hugging Face instead, uncomment and run the cell in Section 2.3.

# 2. Install dependencies (also handled by the first notebook cell)
pip install pypdf langchain-text-splitters sentence-transformers chromadb transformers accelerate datasets

# 3. Run
# Recommended: open in Google Colab. A CPU runtime is sufficient for the
# embedding, generation, and re-ranking models used here.
jupyter notebook Document_QA_RAG_Week7.ipynb
```
Run all cells top to bottom. The first run will download the embedding, generation, and re-ranking models, which takes a couple of minutes; subsequent runs in the same session are much faster.

Evaluation Metrics

| Metric | Formula / Definition | Interpretation |
|---|---|---|
| Retrieval distance | Vector distance between query and retrieved chunk embeddings | Lower distance means a closer semantic match |
| Chunks retrieved | Number of chunks passed into the generation prompt per question | Controls how much context the model has to work with |
| Response time | Wall-clock seconds from question to generated answer | Practical latency of the end-to-end pipeline |
| Hybrid score | Weighted combination of vector similarity and keyword overlap | Balances semantic and exact-match relevance |
| Groundedness (qualitative) | Whether the generated answer is actually supported by the retrieved chunks | Core measure of whether the RAG system is doing its job vs. hallucinating |

Week 7 Assignment — Data Science Internship
