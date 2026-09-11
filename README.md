# DataPilot — AI Data Analyst Agent

**DataPilot** is a portfolio-grade, industry-oriented AI Data Analyst Agent built with Python, FastAPI, LangGraph, LangChain, Pandas/NumPy, RAG, and PostgreSQL. It autonomously ingests structured datasets (CSV, XLSX), inspects schemas, plans analytical workflows, executes code & statistical analysis in a controlled sandbox, generates rich visualizations, and produces executive business insights.

---

## 🎯 Architecture Overview

```
User → Frontend (React/Vite) → FastAPI Backend → LangGraph Orchestrator
                                                        │
         ┌───────────────────┬──────────────────────────┼─────────────────────────┐
         ▼                   ▼                          ▼                         ▼
  Supervisor Agent   Data Profiler Agent      Data Analyst Agent       Visualization Agent
         │                   │                          │                         │
         └───────────────────┴──────────────────────────┴─────────────────────────┘
                                                        │
                                                        ▼
                                           Tools & Controlled Sandbox
                                                        │
                                                        ▼
                                           Business Insights & Response
```

---

## 🚀 Key Features & Capabilities

- **Autonomous Agentic Workflow**: Multi-step reasoning driven by a LangGraph orchestrator.
- **Dataset Profiling & Quality Audit**: Automatic schema detection, data type inference, duplicate tracking, and statistics summaries.
- **Controlled Code Execution**: Sandboxed Python runner preventing unauthorized filesystem/network access.
- **Context-Aware RAG**: Retrieval-Augmented Generation over dataset metadata and business dictionaries.
- **Conversational Memory**: State preservation across multi-turn queries using LangGraph state checkpointers.
- **Production API**: Clean FastAPI asynchronous endpoints backed by Pydantic schemas.

---

## 📁 Project Structure

```
DataPilot/
├── app/
│   ├── api/          # API routes & dependencies
│   ├── agents/       # Specialized agent nodes (Supervisor, Profiler, Analyst, Vis, Insight)
│   ├── graph/        # LangGraph workflow definition & state
│   ├── tools/        # Data analysis, statistics, profiling & visualization tools
│   ├── data/         # Ingestion, validation, loader
│   ├── rag/          # Metadata embeddings & retriever
│   ├── llm/          # Model configuration & prompts
│   ├── database/     # SQLAlchemy models & sessions
│   ├── schemas/      # Pydantic data contracts
│   ├── core/         # Settings, logging & security
│   └── utils/        # Helper functions
├── data/sample/      # Benchmark test datasets
├── tests/            # Automated test suite
├── .env.example      # Environment variables template
├── requirements.txt  # Project dependencies
└── README.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.12+

### 2. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 5. Launch Application
```bash
uvicorn app.main:app --reload
```
Access the API interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
