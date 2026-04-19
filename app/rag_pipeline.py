"""
RAG Pipeline — Groq + HuggingFace Embeddings (100% Free)
"""

import os
import glob
import asyncio
import threading
from typing import AsyncGenerator

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.logger import logger


RAG_PROMPT = PromptTemplate.from_template("""
You are a knowledgeable and helpful assistant. Answer questions accurately
using ONLY the context provided below.

Rules:
- If the answer is not in the context, say "I don't have enough information
  in the provided documents to answer that."
- Always be concise and direct.

Context:
{context}

Question: {question}

Answer:""")


class AsyncTokenCollector(BaseCallbackHandler):
    def __init__(self):
        self.queue: list[str] = []
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.append(token)

    def on_llm_end(self, *args, **kwargs):
        self.done = True


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    PERSIST_DIR = "./chroma_db"
    DATA_DIR = "./data"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    TOP_K = 5
    LLM_MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        logger.info("Loading HuggingFace embeddings (first run downloads ~90MB)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,
        )

    def _load_documents(self, directory: str):
        docs = []
        pdf_files = glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True)
        txt_files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
        md_files  = glob.glob(os.path.join(directory, "**/*.md"),  recursive=True)

        for path in pdf_files:
            try:
                docs.extend(PyPDFLoader(path).load())
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        for path in txt_files + md_files:
            try:
                docs.extend(TextLoader(path, encoding="utf-8").load())
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(docs)} documents")
        return docs

    def _chunk_documents(self, docs):
        chunks = self.splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def _create_sample_doc(self):
        sample_path = os.path.join(self.DATA_DIR, "sample.txt")
        if not os.path.exists(sample_path):
            with open(sample_path, "w") as f:
                f.write("""RAG Chatbot Documentation

This is a production-grade Retrieval-Augmented Generation (RAG) chatbot.

How it works:
RAG combines the power of large language models with a searchable knowledge
base. When you ask a question, the system retrieves the most relevant document
chunks using vector similarity search, then passes those chunks as context to
the LLM to generate a grounded answer.

Key Components:
1. Document Ingestion: PDFs, TXT, and Markdown files are loaded and chunked.
2. Embedding: Each chunk is converted to a vector using HuggingFace
   all-MiniLM-L6-v2 model which runs locally and is completely free.
3. Vector Store: ChromaDB stores and indexes all embeddings.
4. Retrieval: Top-5 most similar chunks retrieved per query using MMR.
5. Generation: Llama3 via Groq generates the final answer using free API.
6. Evaluation: RAGAS metrics measure faithfulness and relevancy.

How to add your own documents:
Use the upload endpoint to add PDF, TXT, or Markdown files to the chatbot.

Evaluation Metrics:
Faithfulness measures if the answer is grounded in the retrieved context.
Answer Relevancy measures if the answer addresses the question asked.
Context Recall measures if the right chunks were retrieved for the query.
""")

    def load_or_build_vectorstore(self):
        if os.path.exists(self.PERSIST_DIR) and os.listdir(self.PERSIST_DIR):
            logger.info("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=self.PERSIST_DIR,
                embedding_function=self.embeddings,
            )
        else:
            logger.info("Building vectorstore from scratch...")
            os.makedirs(self.DATA_DIR, exist_ok=True)
            self._create_sample_doc()
            docs = self._load_documents(self.DATA_DIR)
            chunks = self._chunk_documents(docs)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.PERSIST_DIR,
            )
            logger.info(f"Vectorstore built with {self.vectorstore._collection.count()} vectors")

        self._build_chain()

    def index_file(self, file_path: str) -> int:
        if file_path.endswith(".pdf"):
            docs = PyPDFLoader(file_path).load()
        else:
            docs = TextLoader(file_path, encoding="utf-8").load()
        chunks = self._chunk_documents(docs)
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def _build_chain(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.TOP_K, "fetch_k": 20},
        )
        llm = ChatGroq(
            model=self.LLM_MODEL,
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain built successfully.")

    def query(self, question: str) -> dict:
        # Get answer
        answer = self.chain.invoke(question)

        # Get sources separately
        docs = self.retriever.invoke(question)
        sources = list({
            doc.metadata.get("source", "unknown") for doc in docs
        })

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(docs),
        }

    async def stream_query(self, question: str) -> AsyncGenerator[str, None]:
        collector = AsyncTokenCollector()
        llm = ChatGroq(
            model=self.LLM_MODEL,
            temperature=0,
            streaming=True,
            callbacks=[collector],
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        stream_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        thread = threading.Thread(
            target=lambda: stream_chain.invoke(question)
        )
        thread.start()
        while not collector.done or collector.queue:
            while collector.queue:
                yield collector.queue.pop(0)
            if not collector.done:
                await asyncio.sleep(0.01)
        thread.join()

    def doc_count(self) -> int:
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0

    def evaluate(self) -> dict:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset

            test_questions = [
                "How does the RAG chatbot work?",
                "What embedding model is used?",
                "How do I add my own documents?",
            ]
            rows = []
            for q in test_questions:
                answer = self.chain.invoke(q)
                docs = self.retriever.invoke(q)
                rows.append({
                    "question": q,
                    "answer": answer,
                    "contexts": [d.page_content for d in docs],
                })
            dataset = Dataset.from_list(rows)
            scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
            return {k: round(float(v), 4) for k, v in scores.items()}
        except Exception as e:
            return {"error": str(e)}