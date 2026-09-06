# 🎯 Interview Trainer Agent

> **Problem Statement #22** — A RAG-powered AI interview preparation assistant built on **IBM Granite** via **watsonx.ai**. Generates tailored interview questions, evaluates candidate answers, and builds personalised preparation strategies.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / UI                             │
│              HTML + CSS + Vanilla JS Frontend                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP (REST API)
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                               │
│  /api/questions/generate   /api/answers/evaluate                │
│  /api/strategy/generate    /api/knowledge/ingest                │
│  /api/resume/upload        /health                              │
└─────────┬──────────────────────────────┬────────────────────────┘
          │                              │
┌─────────▼─────────┐        ┌───────────▼────────────────────────┐
│   RAG Pipeline    │        │      IBM Granite on watsonx.ai     │
│                   │        │                                    │
│  ChromaDB         │        │  ibm/granite-3-8b-instruct         │
│  (Vector Store)   │        │  (ibm-watsonx-ai SDK)              │
│                   │        │                                    │
│  sentence-        │        │  Structured JSON prompts for:      │
│  transformers     │        │  • Question generation             │
│  (Embeddings)     │        │  • Answer evaluation               │
│                   │        │  • Preparation strategy            │
│  Knowledge Base:  │        │                                    │
│  • Behavioral     │        │  Prompt templates use              │
│  • Technical      │        │  <|system|> / <|user|> /           │
│  • HR Interview   │        │  <|assistant|> Granite format      │
│  • Job Roles      │        │                                    │
│  • Soft Skills    │        └────────────────────────────────────┘
│  + Custom Docs    │
└───────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Tailored Question Generation** | Generates role-specific, experience-adjusted questions across 6 categories |
| **Answer Evaluation** | Scores answers 0–10 with strengths, improvements, and model answers |
| **Preparation Strategy** | Day-by-day study plan, key topics, resources, and confidence tips |
| **RAG Knowledge Base** | Pre-seeded with 7 expert documents; accepts custom job descriptions & guides |
| **Resume Upload** | Ingests PDF/DOCX/TXT resumes for personalised question generation |
| **IBM Granite** | Uses `ibm/granite-3-8b-instruct` with structured prompt templates |
| **Interactive UI** | Clean single-page frontend with practice mode and instant feedback |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- IBM Cloud account with watsonx.ai access ([sign up free](https://cloud.ibm.com/registration))
- A watsonx.ai project with an API key

### 2. Clone and install

```bash
git clone <repo-url>
cd interview-trainer-agent
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env with your watsonx.ai API key and Project ID
```

Required values in `.env`:
```
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 4. Run

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

---

## 🐳 Docker

```bash
# Build
docker build -t interview-trainer-agent .

# Run (with env file)
docker run -p 8000:8000 --env-file .env interview-trainer-agent
```

---

## 📡 API Reference

### `GET /health`
Returns service health, model ID, and knowledge base chunk count.

### `POST /api/questions/generate`
Generate tailored interview questions.

```json
{
  "profile": {
    "name": "Priya Sharma",
    "job_role": "Senior Data Engineer",
    "experience_level": "senior",
    "skills": ["Python", "Spark", "AWS", "SQL"],
    "industry": "FinTech",
    "resume_text": "Optional resume paste..."
  },
  "categories": ["technical", "behavioral", "system_design"],
  "num_questions": 10
}
```

### `POST /api/answers/evaluate`
Evaluate a candidate's answer with score and detailed feedback.

```json
{
  "profile": { "name": "...", "job_role": "...", "experience_level": "mid" },
  "question": "Describe a time you optimised a slow query.",
  "user_answer": "I identified a missing index on a join column...",
  "question_category": "technical"
}
```

### `POST /api/strategy/generate`
Build a personalised preparation plan.

```json
{
  "profile": { "name": "...", "job_role": "...", "experience_level": "junior" },
  "weak_areas": ["System design", "Communication"],
  "interview_date_days": 7
}
```

### `POST /api/knowledge/ingest`
Add a custom document to the RAG knowledge base.

```json
{
  "content": "Full document text...",
  "source": "IBM Cloud Job Description 2024",
  "doc_type": "job_description"
}
```

### `POST /api/resume/upload`
Upload a PDF, DOCX, or TXT resume file (multipart form).

---

## 🧪 Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📁 Project Structure

```
interview-trainer-agent/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings from .env
│   ├── models.py          # Pydantic request/response schemas
│   ├── rag.py             # ChromaDB vector store + document ingestion
│   ├── granite.py         # IBM Granite watsonx.ai client + prompts
│   ├── agent.py           # Orchestration: RAG + LLM + parsing
│   └── main.py            # FastAPI application + routes
├── frontend/
│   └── index.html         # Single-page UI
├── tests/
│   └── test_agent.py      # Unit + integration tests
├── data/
│   └── chroma_db/         # ChromaDB persistence (auto-created)
├── .env.example
├── requirements.txt
├── Dockerfile
└── run.py
```

---

## 🤖 IBM Granite Prompt Design

Each agent task uses structured Granite-format prompts:

```
<|system|>
You are an expert interview coach...
<|user|>
Candidate: {name}
Role: {job_role}
...
Return JSON only.
<|assistant|>
```

This ensures the model produces clean, structured JSON that the agent layer parses and validates into Pydantic response models. Fallback stubs activate if JSON parsing fails, ensuring graceful degradation.

---

## 🌐 IBM Cloud Services Used

| Service | Usage |
|---|---|
| **IBM watsonx.ai** | Hosts IBM Granite 3 8B Instruct model for generation |
| **IBM Granite 3 8B Instruct** | Core LLM for question generation, evaluation, and strategy |

> **IBM Cloud Lite tier compatible** — watsonx.ai offers a free tier sufficient for development and demo workloads.

---

## 📝 License

MIT License — built for IBM hackathon Problem Statement #22.
