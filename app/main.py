"""
FastAPI application – Interview Trainer Agent API.

Endpoints:
  GET  /health                    – service health check
  POST /api/questions/generate    – generate tailored interview questions
  POST /api/answers/evaluate      – evaluate a candidate's answer
  POST /api/strategy/generate     – generate a preparation strategy
  POST /api/knowledge/ingest      – ingest a new document into the RAG store
  POST /api/resume/upload         – upload a PDF/DOCX resume and ingest it
"""

from __future__ import annotations

import io
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import agent, rag
from app.config import get_settings
from app.models import (
    EvaluateAnswerRequest,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    AnswerEvaluation,
    HealthResponse,
    IngestDocumentRequest,
    IngestResponse,
    PreparationStrategy,
    PreparationStrategyRequest,
)

# ─── Logging ──────────────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Interview Trainer Agent …")
    # Seed the knowledge base with built-in documents
    try:
        count = rag.seed_knowledge_base()
        logger.info("Knowledge base ready. Total chunks: %d", count)
    except Exception as exc:
        logger.error("Knowledge base seeding failed: %s", exc)
    yield
    logger.info("Interview Trainer Agent shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Trainer Agent",
    description=(
        "RAG-powered AI interview preparation assistant using IBM Granite on watsonx.ai. "
        "Generates tailored questions, evaluates answers, and builds personalised prep strategies."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Returns service health status and knowledge base statistics."""
    return HealthResponse(
        status="ok",
        granite_model=settings.granite_model_id,
        vector_store_documents=rag.get_collection_count(),
    )


@app.post(
    "/api/questions/generate",
    response_model=GenerateQuestionsResponse,
    tags=["Interview Questions"],
    summary="Generate tailored interview questions",
)
async def generate_questions(req: GenerateQuestionsRequest):
    """
    Generate a personalised set of interview questions based on:
    - Candidate profile (name, role, experience level, skills)
    - Desired question categories (technical, behavioral, HR, etc.)
    - Optional resume text for further personalisation
    """
    try:
        result = agent.generate_questions(
            profile=req.profile,
            categories=req.categories,
            num_questions=req.num_questions,
        )
        return result
    except Exception as exc:
        logger.exception("Error generating questions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/answers/evaluate",
    response_model=AnswerEvaluation,
    tags=["Answer Evaluation"],
    summary="Evaluate a candidate's interview answer",
)
async def evaluate_answer(req: EvaluateAnswerRequest):
    """
    Evaluates the quality of a candidate's answer to an interview question.
    Returns a score (0-10), strengths, improvement areas, and a model answer.
    """
    try:
        result = agent.evaluate_answer(
            profile=req.profile,
            question=req.question,
            user_answer=req.user_answer,
            question_category=req.question_category,
        )
        return result
    except Exception as exc:
        logger.exception("Error evaluating answer")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/strategy/generate",
    response_model=PreparationStrategy,
    tags=["Preparation Strategy"],
    summary="Generate a personalised interview preparation strategy",
)
async def generate_strategy(req: PreparationStrategyRequest):
    """
    Builds a day-by-day study plan, key topics, resources, and confidence tips
    tailored to the candidate's role, experience, and time until the interview.
    """
    try:
        result = agent.generate_preparation_strategy(
            profile=req.profile,
            weak_areas=req.weak_areas or [],
            interview_date_days=req.interview_date_days or 7,
        )
        return result
    except Exception as exc:
        logger.exception("Error generating strategy")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/knowledge/ingest",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
    summary="Ingest a document into the RAG knowledge base",
)
async def ingest_document(req: IngestDocumentRequest):
    """
    Add a new document (job description, company guidelines, interview notes)
    to the vector knowledge base for retrieval.
    """
    try:
        chunks = rag.ingest_document(
            content=req.content,
            source=req.source,
            doc_type=req.doc_type,
        )
        return IngestResponse(
            success=True,
            message=f"Successfully ingested document from '{req.source}'.",
            document_chunks=chunks,
        )
    except Exception as exc:
        logger.exception("Error ingesting document")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/resume/upload",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
    summary="Upload a resume file (PDF or DOCX) to enrich the knowledge base",
)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX resume. The text is extracted and ingested into
    the knowledge base for use in personalised question generation.
    """
    filename = file.filename or "resume"
    content_type = file.content_type or ""

    try:
        raw_bytes = await file.read()

        if "pdf" in content_type or filename.lower().endswith(".pdf"):
            text = _extract_pdf_text(raw_bytes)
        elif "word" in content_type or filename.lower().endswith(".docx"):
            text = _extract_docx_text(raw_bytes)
        elif filename.lower().endswith(".txt"):
            text = raw_bytes.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type. Please upload a PDF, DOCX, or TXT file.",
            )

        chunks = rag.ingest_document(content=text, source=filename, doc_type="resume")
        return IngestResponse(
            success=True,
            message=f"Resume '{filename}' ingested successfully.",
            document_chunks=chunks,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing resume upload")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── File parsers ─────────────────────────────────────────────────────────────

def _extract_pdf_text(data: bytes) -> str:
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx_text(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n".join(para.text for para in doc.paragraphs)
