"""
Pydantic models for all API request/response schemas.
"""

from __future__ import annotations
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class ExperienceLevel(str, Enum):
    FRESHER = "fresher"
    JUNIOR = "junior"          # 1–3 yrs
    MID = "mid"                # 3–6 yrs
    SENIOR = "senior"          # 6–10 yrs
    LEAD = "lead"              # 10+ yrs


class QuestionCategory(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"
    HR = "hr"
    SYSTEM_DESIGN = "system_design"
    CODING = "coding"


# ─── Request Models ───────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Candidate's full name")
    job_role: str = Field(..., min_length=2, max_length=100, description="Target job role (e.g. 'Senior Python Developer')")
    experience_level: ExperienceLevel = Field(..., description="Candidate's experience level")
    skills: Optional[List[str]] = Field(default=[], description="Key skills (e.g. ['Python', 'AWS', 'Kubernetes'])")
    industry: Optional[str] = Field(default="Technology", description="Target industry")
    resume_text: Optional[str] = Field(default=None, description="Paste resume text for personalized questions")


class GenerateQuestionsRequest(BaseModel):
    profile: UserProfile
    categories: Optional[List[QuestionCategory]] = Field(
        default=[QuestionCategory.TECHNICAL, QuestionCategory.BEHAVIORAL, QuestionCategory.HR],
        description="Question categories to generate"
    )
    num_questions: Optional[int] = Field(default=10, ge=3, le=30, description="Total questions to generate")


class EvaluateAnswerRequest(BaseModel):
    profile: UserProfile
    question: str = Field(..., description="The interview question that was asked")
    user_answer: str = Field(..., min_length=10, description="The candidate's answer to evaluate")
    question_category: QuestionCategory = Field(default=QuestionCategory.BEHAVIORAL)


class PreparationStrategyRequest(BaseModel):
    profile: UserProfile
    weak_areas: Optional[List[str]] = Field(default=[], description="Areas the user wants to improve")
    interview_date_days: Optional[int] = Field(default=7, ge=1, le=90, description="Days until interview")


class IngestDocumentRequest(BaseModel):
    content: str = Field(..., description="Raw text content of the document to ingest")
    source: str = Field(..., description="Source name/URL of the document")
    doc_type: str = Field(default="general", description="Document type: job_description | interview_guide | company_info | general")


# ─── Response Models ──────────────────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    question: str
    category: QuestionCategory
    difficulty: str                    # easy | medium | hard
    model_answer: str
    tips: List[str]


class GenerateQuestionsResponse(BaseModel):
    profile_name: str
    job_role: str
    experience_level: str
    questions: List[InterviewQuestion]
    rag_sources: List[str] = Field(default=[], description="Knowledge base sources used")


class AnswerEvaluation(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score out of 10")
    strengths: List[str]
    improvements: List[str]
    model_answer: str
    detailed_feedback: str


class PreparationStrategy(BaseModel):
    profile_name: str
    job_role: str
    study_plan: List[dict]             # [{day, focus, tasks[]}]
    key_topics: List[str]
    recommended_resources: List[str]
    confidence_tips: List[str]
    rag_sources: List[str] = Field(default=[])


class IngestResponse(BaseModel):
    success: bool
    message: str
    document_chunks: int


class HealthResponse(BaseModel):
    status: str
    granite_model: str
    vector_store_documents: int
    version: str = "1.0.0"
