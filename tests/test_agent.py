"""
Tests for the Interview Trainer Agent.

Run:  pytest tests/ -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ─── RAG pipeline tests ───────────────────────────────────────────────────────

class TestRAGPipeline:
    """Test document ingestion and retrieval."""

    def test_chunk_text_basic(self):
        from app.rag import _chunk_text
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_short(self):
        from app.rag import _chunk_text
        text = "Short text"
        chunks = _chunk_text(text, chunk_size=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_overlap(self):
        from app.rag import _chunk_text
        words = [f"w{i}" for i in range(60)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        # Each chunk should not exceed chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 20


# ─── Agent safe JSON parse ────────────────────────────────────────────────────

class TestSafeJsonParse:
    """Test the JSON extraction helper."""

    def test_plain_json_array(self):
        from app.agent import _safe_json_parse
        raw = '[{"question": "Tell me about yourself"}]'
        result = _safe_json_parse(raw)
        assert isinstance(result, list)
        assert result[0]["question"] == "Tell me about yourself"

    def test_json_with_markdown_fence(self):
        from app.agent import _safe_json_parse
        raw = '```json\n[{"score": 7}]\n```'
        result = _safe_json_parse(raw)
        assert result[0]["score"] == 7

    def test_json_with_preamble(self):
        from app.agent import _safe_json_parse
        raw = 'Here is the JSON:\n{"score": 8, "strengths": ["good"]}'
        result = _safe_json_parse(raw)
        assert result["score"] == 8

    def test_invalid_json_raises(self):
        from app.agent import _safe_json_parse
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _safe_json_parse("not json at all ...")


# ─── Models validation ────────────────────────────────────────────────────────

class TestModels:
    """Test Pydantic model validation."""

    def test_user_profile_valid(self):
        from app.models import UserProfile, ExperienceLevel
        profile = UserProfile(
            name="Alice",
            job_role="Backend Engineer",
            experience_level=ExperienceLevel.MID,
        )
        assert profile.name == "Alice"
        assert profile.skills == []

    def test_generate_request_defaults(self):
        from app.models import GenerateQuestionsRequest, UserProfile, ExperienceLevel, QuestionCategory
        req = GenerateQuestionsRequest(
            profile=UserProfile(name="Bob", job_role="ML Engineer", experience_level=ExperienceLevel.JUNIOR)
        )
        assert req.num_questions == 10
        assert QuestionCategory.TECHNICAL in req.categories

    def test_evaluate_request_min_answer_length(self):
        from app.models import EvaluateAnswerRequest, UserProfile, ExperienceLevel
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvaluateAnswerRequest(
                profile=UserProfile(name="X", job_role="Eng", experience_level=ExperienceLevel.FRESHER),
                question="Why this role?",
                user_answer="ok",   # Too short
            )

    def test_ingest_request(self):
        from app.models import IngestDocumentRequest
        req = IngestDocumentRequest(content="Long document text here.", source="test_source")
        assert req.doc_type == "general"


# ─── API endpoint tests (mocked) ─────────────────────────────────────────────

class TestAPIEndpoints:
    """Integration-style tests using FastAPI TestClient with mocked Granite calls."""

    @pytest.fixture(autouse=True)
    def mock_rag_and_granite(self):
        """Patch out external calls so tests run without credentials."""
        with (
            patch('app.rag._get_collection') as mock_col,
            patch('app.rag._get_embedder') as mock_emb,
            patch('app.granite._get_model') as mock_model,
        ):
            # Mock collection
            mock_collection = MagicMock()
            mock_collection.count.return_value = 42
            mock_collection.query.return_value = {
                "documents": [["Context about interviews"]],
                "metadatas": [[{"source": "test_source", "doc_type": "interview_guide"}]],
                "distances": [[0.1]],
            }
            mock_col.return_value = mock_collection

            # Mock embedder
            import numpy as np
            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.array([[0.1] * 384])
            mock_emb.return_value = mock_embedder

            # Mock Granite model
            mock_granite = MagicMock()
            mock_granite.generate_text.return_value = json.dumps([
                {
                    "question": "Describe your experience with Python.",
                    "category": "technical",
                    "difficulty": "medium",
                    "model_answer": "I have 5 years of Python experience...",
                    "tips": ["Give specific examples", "Mention libraries"],
                }
            ])
            mock_model.return_value = mock_granite

            yield

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with patch('app.rag.seed_knowledge_base', return_value=100):
            client = TestClient(app)
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ok"
            assert "granite_model" in data

    def test_generate_questions_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with patch('app.rag.seed_knowledge_base', return_value=100):
            client = TestClient(app)
            res = client.post("/api/questions/generate", json={
                "profile": {
                    "name": "Test User",
                    "job_role": "Python Developer",
                    "experience_level": "mid",
                },
                "categories": ["technical"],
                "num_questions": 1,
            })
            assert res.status_code == 200
            data = res.json()
            assert data["profile_name"] == "Test User"
            assert isinstance(data["questions"], list)

    def test_generate_strategy_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        mock_strategy = json.dumps({
            "study_plan": [{"day": 1, "focus": "Review Python", "tasks": ["Read docs", "Practice"]}],
            "key_topics": ["Python", "SQL"],
            "recommended_resources": ["LeetCode"],
            "confidence_tips": ["Sleep well before the interview"],
        })
        with (
            patch('app.rag.seed_knowledge_base', return_value=100),
            patch('app.granite._get_model') as mock_model,
        ):
            m = MagicMock()
            m.generate_text.return_value = mock_strategy
            mock_model.return_value = m
            client = TestClient(app)
            res = client.post("/api/strategy/generate", json={
                "profile": {
                    "name": "Jane",
                    "job_role": "Data Scientist",
                    "experience_level": "junior",
                },
                "weak_areas": ["Statistics"],
                "interview_date_days": 5,
            })
            assert res.status_code == 200
            data = res.json()
            assert data["profile_name"] == "Jane"

    def test_ingest_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        with (
            patch('app.rag.seed_knowledge_base', return_value=100),
            patch('app.rag.ingest_document', return_value=3),
        ):
            client = TestClient(app)
            res = client.post("/api/knowledge/ingest", json={
                "content": "A" * 200,
                "source": "test_doc",
                "doc_type": "general",
            })
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
