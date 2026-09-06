"""
IBM Granite LLM Client via watsonx.ai.

Wraps ibm-watsonx-ai SDK with retry logic and structured prompt templates
for each agent task: question generation, answer evaluation, and prep strategy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Lazy client initialisation ──────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            credentials = {
                "url": settings.watsonx_url,
                "apikey": settings.watsonx_api_key,
            }

            params = {
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.MIN_NEW_TOKENS: 20,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            }

            _model = ModelInference(
                model_id=settings.granite_model_id,
                credentials=credentials,
                project_id=settings.watsonx_project_id,
                params=params,
            )
            logger.info("IBM Granite model initialised: %s", settings.granite_model_id)
        except Exception as exc:
            logger.error("Failed to initialise IBM Granite model: %s", exc)
            raise
    return _model


# ─── Core generation with retry ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _generate(prompt: str) -> str:
    model = _get_model()
    response = model.generate_text(prompt=prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()


# ─── Prompt templates ─────────────────────────────────────────────────────────

def _build_question_prompt(
    name: str,
    job_role: str,
    experience_level: str,
    skills: List[str],
    industry: str,
    categories: List[str],
    num_questions: int,
    context: str,
    resume_snippet: Optional[str],
) -> str:
    skills_str = ", ".join(skills) if skills else "general technical skills"
    categories_str = ", ".join(categories)
    resume_section = f"\nResume highlights:\n{resume_snippet[:800]}\n" if resume_snippet else ""

    return f"""<|system|>
You are an expert interview coach and hiring manager with 15 years of experience across the technology industry.
Your role is to generate highly relevant, tailored interview questions for job candidates.
Always respond with valid JSON only — no explanation, no markdown fences, just the raw JSON array.
<|user|>
Generate exactly {num_questions} interview questions for the following candidate:

Candidate Name: {name}
Target Role: {job_role}
Experience Level: {experience_level}
Key Skills: {skills_str}
Industry: {industry}
Question Categories: {categories_str}
{resume_section}
Relevant knowledge base context:
{context}

Return a JSON array of exactly {num_questions} objects. Each object must have:
- "question": string (the interview question)
- "category": one of [{categories_str}]
- "difficulty": one of ["easy", "medium", "hard"]
- "model_answer": string (a concise, strong model answer, 3-5 sentences)
- "tips": array of 2-3 strings (quick tips for answering this specific question)

Ensure questions cover different difficulties and are specific to the {job_role} role at {experience_level} level.
JSON:
<|assistant|>"""


def _build_evaluation_prompt(
    name: str,
    job_role: str,
    experience_level: str,
    question: str,
    user_answer: str,
    category: str,
    context: str,
) -> str:
    return f"""<|system|>
You are a strict but constructive interview evaluator. Evaluate candidate answers objectively.
Respond with valid JSON only — no explanation, no markdown.
<|user|>
Evaluate this interview answer for:

Candidate: {name}
Role: {job_role}
Experience Level: {experience_level}
Category: {category}

Question: {question}

Candidate's Answer: {user_answer}

Relevant context:
{context}

Return a JSON object with:
- "score": integer 0-10 (10 = perfect answer)
- "strengths": array of 2-3 specific strengths in the answer
- "improvements": array of 2-3 specific areas to improve
- "model_answer": string (an ideal answer to this question, 4-6 sentences)
- "detailed_feedback": string (2-3 paragraph constructive feedback)

Be honest, specific, and actionable. Score strictly based on {experience_level} expectations.
JSON:
<|assistant|>"""


def _build_strategy_prompt(
    name: str,
    job_role: str,
    experience_level: str,
    skills: List[str],
    industry: str,
    weak_areas: List[str],
    days_until_interview: int,
    context: str,
) -> str:
    skills_str = ", ".join(skills) if skills else "core role skills"
    weak_str = ", ".join(weak_areas) if weak_areas else "general preparation"

    return f"""<|system|>
You are a career coach specialising in interview preparation and job search strategy.
Respond with valid JSON only — no explanation, no markdown fences.
<|user|>
Create a comprehensive interview preparation strategy for:

Candidate: {name}
Target Role: {job_role}
Experience Level: {experience_level}
Industry: {industry}
Current Skills: {skills_str}
Areas to Improve: {weak_str}
Days Until Interview: {days_until_interview}

Knowledge base context:
{context}

Return a JSON object with:
- "study_plan": array of objects, one per day (up to {min(days_until_interview, 7)} days), each with:
  - "day": integer
  - "focus": string (main theme for the day)
  - "tasks": array of 3-4 specific task strings
- "key_topics": array of 8-10 strings (most important topics to master)
- "recommended_resources": array of 5-7 strings (books, websites, platforms, courses)
- "confidence_tips": array of 5 strings (psychological and practical confidence-building tips)

Make the plan realistic and specific to {job_role} at {experience_level} level.
JSON:
<|assistant|>"""


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_questions(
    name: str,
    job_role: str,
    experience_level: str,
    skills: List[str],
    industry: str,
    categories: List[str],
    num_questions: int,
    context: str,
    resume_snippet: Optional[str] = None,
) -> str:
    prompt = _build_question_prompt(
        name, job_role, experience_level, skills, industry,
        categories, num_questions, context, resume_snippet
    )
    return _generate(prompt)


def evaluate_answer(
    name: str,
    job_role: str,
    experience_level: str,
    question: str,
    user_answer: str,
    category: str,
    context: str,
) -> str:
    prompt = _build_evaluation_prompt(
        name, job_role, experience_level, question, user_answer, category, context
    )
    return _generate(prompt)


def generate_preparation_strategy(
    name: str,
    job_role: str,
    experience_level: str,
    skills: List[str],
    industry: str,
    weak_areas: List[str],
    days_until_interview: int,
    context: str,
) -> str:
    prompt = _build_strategy_prompt(
        name, job_role, experience_level, skills, industry,
        weak_areas, days_until_interview, context
    )
    return _generate(prompt)


def get_model_id() -> str:
    return settings.granite_model_id
