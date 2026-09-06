"""
Agent logic layer – orchestrates RAG retrieval + IBM Granite generation
and parses/validates structured JSON responses.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from app import granite, rag
from app.models import (
    AnswerEvaluation,
    ExperienceLevel,
    GenerateQuestionsResponse,
    InterviewQuestion,
    PreparationStrategy,
    QuestionCategory,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json_parse(raw: str) -> any:
    """
    Attempt to parse JSON from Granite output.
    Handles cases where model wraps output in markdown fences.
    """
    text = raw.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find first [ or { to handle preamble text
    for start_char in ("[", "{"):
        idx = text.find(start_char)
        if idx != -1:
            text = text[idx:]
            break
    return json.loads(text)


def _rag_query_for_profile(profile: UserProfile, extra: str = "") -> dict:
    query = f"{profile.job_role} {profile.experience_level.value} {' '.join(profile.skills or [])} {extra}"
    return rag.retrieve_context(query.strip())


def _format_context(documents: List[str]) -> str:
    if not documents:
        return "No additional context available."
    return "\n\n---\n\n".join(documents[:5])


# ─── Question Generation ──────────────────────────────────────────────────────

def generate_questions(
    profile: UserProfile,
    categories: List[QuestionCategory],
    num_questions: int,
) -> GenerateQuestionsResponse:
    # Retrieve relevant RAG context
    rag_result = _rag_query_for_profile(profile, "interview questions")
    context = _format_context(rag_result["documents"])

    cat_strs = [c.value for c in categories]

    raw = granite.generate_questions(
        name=profile.name,
        job_role=profile.job_role,
        experience_level=profile.experience_level.value,
        skills=profile.skills or [],
        industry=profile.industry or "Technology",
        categories=cat_strs,
        num_questions=num_questions,
        context=context,
        resume_snippet=profile.resume_text,
    )

    try:
        questions_data = _safe_json_parse(raw)
        if not isinstance(questions_data, list):
            questions_data = questions_data.get("questions", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error("JSON parse error in generate_questions: %s\nRaw: %s", exc, raw[:500])
        questions_data = _fallback_questions(profile, categories, num_questions)

    questions: List[InterviewQuestion] = []
    for item in questions_data:
        try:
            questions.append(
                InterviewQuestion(
                    question=item.get("question", ""),
                    category=QuestionCategory(item.get("category", "behavioral")),
                    difficulty=item.get("difficulty", "medium"),
                    model_answer=item.get("model_answer", ""),
                    tips=item.get("tips", []),
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed question item: %s", exc)

    return GenerateQuestionsResponse(
        profile_name=profile.name,
        job_role=profile.job_role,
        experience_level=profile.experience_level.value,
        questions=questions,
        rag_sources=rag_result["sources"],
    )


# ─── Answer Evaluation ────────────────────────────────────────────────────────

def evaluate_answer(
    profile: UserProfile,
    question: str,
    user_answer: str,
    question_category: QuestionCategory,
) -> AnswerEvaluation:
    rag_result = _rag_query_for_profile(profile, question)
    context = _format_context(rag_result["documents"])

    raw = granite.evaluate_answer(
        name=profile.name,
        job_role=profile.job_role,
        experience_level=profile.experience_level.value,
        question=question,
        user_answer=user_answer,
        category=question_category.value,
        context=context,
    )

    try:
        data = _safe_json_parse(raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error("JSON parse error in evaluate_answer: %s\nRaw: %s", exc, raw[:500])
        data = _fallback_evaluation()

    return AnswerEvaluation(
        score=int(data.get("score", 5)),
        strengths=data.get("strengths", ["Answer addresses the question"]),
        improvements=data.get("improvements", ["Add more specific examples"]),
        model_answer=data.get("model_answer", ""),
        detailed_feedback=data.get("detailed_feedback", ""),
    )


# ─── Preparation Strategy ─────────────────────────────────────────────────────

def generate_preparation_strategy(
    profile: UserProfile,
    weak_areas: List[str],
    interview_date_days: int,
) -> PreparationStrategy:
    rag_result = _rag_query_for_profile(profile, "preparation strategy study plan")
    context = _format_context(rag_result["documents"])

    raw = granite.generate_preparation_strategy(
        name=profile.name,
        job_role=profile.job_role,
        experience_level=profile.experience_level.value,
        skills=profile.skills or [],
        industry=profile.industry or "Technology",
        weak_areas=weak_areas,
        days_until_interview=interview_date_days,
        context=context,
    )

    try:
        data = _safe_json_parse(raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error("JSON parse error in strategy: %s\nRaw: %s", exc, raw[:500])
        data = _fallback_strategy(profile, interview_date_days)

    return PreparationStrategy(
        profile_name=profile.name,
        job_role=profile.job_role,
        study_plan=data.get("study_plan", []),
        key_topics=data.get("key_topics", []),
        recommended_resources=data.get("recommended_resources", []),
        confidence_tips=data.get("confidence_tips", []),
        rag_sources=rag_result["sources"],
    )


# ─── Fallback stubs (used when LLM output is unparseable) ────────────────────

def _fallback_questions(profile, categories, num_questions):
    """Return generic fallback questions when JSON parse fails."""
    base = [
        {"question": f"Walk me through your background as a {profile.job_role}.",
         "category": "hr", "difficulty": "easy",
         "model_answer": "I have worked in... (describe your journey concisely)",
         "tips": ["Keep it under 2 minutes", "End with why you want this role"]},
        {"question": "Describe a challenging project and how you overcame obstacles.",
         "category": "behavioral", "difficulty": "medium",
         "model_answer": "Using the STAR method: Situation → Task → Action → Result...",
         "tips": ["Use STAR format", "Quantify the result", "Focus on YOUR actions"]},
        {"question": "Where do you see yourself in 5 years?",
         "category": "hr", "difficulty": "easy",
         "model_answer": "I aim to grow into a senior/lead role...",
         "tips": ["Align with company growth", "Show ambition but be realistic"]},
    ]
    return base[:num_questions]


def _fallback_evaluation():
    return {
        "score": 5,
        "strengths": ["You addressed the question directly"],
        "improvements": ["Add concrete examples using STAR format", "Quantify your impact with metrics"],
        "model_answer": "A strong answer would use the STAR method with specific examples.",
        "detailed_feedback": "Your answer shows some relevant experience. To strengthen it, add specific examples with measurable outcomes.",
    }


def _fallback_strategy(profile, days):
    return {
        "study_plan": [
            {"day": i + 1, "focus": f"Day {i+1} preparation",
             "tasks": ["Review core concepts", "Practice 2 questions", "Research company"]}
            for i in range(min(days, 7))
        ],
        "key_topics": [f"{profile.job_role} fundamentals", "Behavioral questions", "System design",
                       "Company research", "Salary negotiation"],
        "recommended_resources": ["LeetCode", "Glassdoor", "LinkedIn Learning", "System Design Primer (GitHub)",
                                  "Cracking the Coding Interview"],
        "confidence_tips": ["Practice mock interviews aloud", "Get 8 hours of sleep the night before",
                            "Prepare your questions for the interviewer", "Dress professionally",
                            "Arrive 10 minutes early"],
    }
