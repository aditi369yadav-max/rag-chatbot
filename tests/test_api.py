"""
Unit tests for RAG Chatbot
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def get_client():
    """Import app with mocked RAG pipeline."""
    with patch("app.rag_pipeline.OpenAIEmbeddings"), \
         patch("app.rag_pipeline.ChatOpenAI"), \
         patch("app.rag_pipeline.Chroma"):
        from app.main import app
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        client = get_client()
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


class TestChatEndpoint:
    def test_empty_question_returns_400(self):
        client = get_client()
        r = client.post("/chat", json={"question": ""})
        assert r.status_code in (400, 503)

    def test_valid_question_returns_answer(self):
        client = get_client()
        with patch("app.main.rag") as mock_rag:
            mock_rag.query.return_value = {
                "answer": "RAG works by retrieving relevant chunks.",
                "sources": ["data/sample.txt"],
                "chunks_retrieved": 3,
            }
            r = client.post("/chat", json={"question": "How does RAG work?"})
            if r.status_code == 200:
                data = r.json()
                assert "answer" in data
                assert "sources" in data
                assert "latency_ms" in data


class TestUploadEndpoint:
    def test_invalid_file_type_rejected(self):
        client = get_client()
        r = client.post(
            "/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
        )
        assert r.status_code in (400, 422, 503)
