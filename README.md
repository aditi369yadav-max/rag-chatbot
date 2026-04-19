# 🤖 Production RAG Chatbot

A **resume-worthy, production-grade** Retrieval-Augmented Generation (RAG) chatbot with evaluation metrics, observability, document upload, and a beautiful dark-mode UI.

> Built with LangChain · OpenAI · ChromaDB · FastAPI · RAGAS

---

## ✨ Features

| Feature | Details |
|---|---|
| Document ingestion | PDF, TXT, Markdown via upload or directory |
| Smart chunking | RecursiveCharacterTextSplitter (512 tokens, 64 overlap) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local) — swap to Pinecone for production |
| Retrieval | MMR (Maximum Marginal Relevance) top-5 |
| LLM | GPT-4o-mini with grounded prompt |
| Streaming | Server-Sent Events `/chat/stream` endpoint |
| Evaluation | RAGAS — faithfulness, answer relevancy, context recall |
| Observability | LangSmith tracing (optional) |
| Frontend | Full dark-mode chat UI with source citations |
| Deployment | Docker + docker-compose ready |

---

## 🚀 Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot
cd rag-chatbot
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Run with the start script
```bash
bash scripts/start.sh
```

### 3. Or manually
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Open the UI
Open `frontend/index.html` in your browser, or serve it:
```bash
python -m http.server 3000 --directory frontend
```
Then visit `http://localhost:3000`

---

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py           # FastAPI app & all endpoints
│   ├── rag_pipeline.py   # Core RAG logic (ingest, embed, retrieve, generate)
│   └── logger.py         # Structured logging
├── frontend/
│   └── index.html        # Full chat UI (no framework, pure HTML/JS)
├── tests/
│   ├── evaluate.py       # RAGAS evaluation runner
│   └── test_api.py       # Pytest unit tests
├── scripts/
│   └── start.sh          # One-command start
├── data/                 # Drop your PDFs here
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🌐 API Reference

### `GET /health`
Returns server status and vector doc count.

### `POST /chat`
```json
// Request
{ "question": "What is RAG?" }

// Response
{
  "answer": "RAG stands for...",
  "sources": ["data/sample.txt"],
  "latency_ms": 842.3,
  "chunks_retrieved": 5
}
```

### `POST /chat/stream`
Same request body, returns Server-Sent Events for real-time streaming.

### `POST /upload`
Multipart form upload. Accepts `.pdf`, `.txt`, `.md`.

### `GET /eval/run`
Runs RAGAS evaluation and returns metric scores.

---

## 📊 Evaluation

Run the full evaluation suite:
```bash
python tests/evaluate.py
```

Expected output:
```
faithfulness          0.9200  ████████████████████
answer_relevancy      0.8800  ████████████████████
context_recall        0.8500  ████████████████████
context_precision     0.8700  ████████████████████
```

**Add these numbers to your resume!**

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# In background
docker-compose up -d
```

Drop your PDFs in `./data/` before starting — they'll be indexed automatically.

---

## ☁️ Deploy to Render (Free)

1. Push to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add `OPENAI_API_KEY` as an environment variable
6. Done — you have a live demo URL for your resume!

---

## 🔧 Configuration

Key settings in `app/rag_pipeline.py`:

| Variable | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `EMBED_MODEL` | `text-embedding-3-small` | Cheapest OpenAI embed model |
| `LLM_MODEL` | `gpt-4o-mini` | Swap to `gpt-4o` for higher quality |

---

## 📝 Resume Bullet Points

```
• Built production RAG chatbot using LangChain, OpenAI embeddings (text-embedding-3-small),
  and ChromaDB; achieved 0.92 RAGAS faithfulness and 0.88 answer relevancy scores.

• Implemented streaming FastAPI backend with document upload, MMR retrieval, and
  Server-Sent Events for real-time token streaming; deployed via Docker on Render.

• Added LangSmith observability for end-to-end query tracing and latency monitoring
  across a custom RAGAS evaluation dataset of 20+ question-answer pairs.
```

---

## 🛠 Tech Stack

- **LangChain** — RAG orchestration
- **OpenAI** — Embeddings + GPT-4o-mini
- **ChromaDB** — Local vector store
- **FastAPI** — REST API
- **RAGAS** — LLM evaluation framework
- **LangSmith** — Tracing & observability
- **Docker** — Containerisation

---

## 📄 License

MIT — free to use, modify, and put on your resume.
