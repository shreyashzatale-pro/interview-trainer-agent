"""
RAG Pipeline – Document ingestion, embedding, and retrieval using ChromaDB.

Supports:
  - Ingesting raw text documents into the vector store
  - Seeding the knowledge base with built-in interview resources
  - Retrieving relevant context for a given query
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ─── Singleton vector store ───────────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready. Documents: %d",
            settings.chroma_collection_name,
            _collection.count(),
        )
    return _collection


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─── Ingest ───────────────────────────────────────────────────────────────────

def ingest_document(content: str, source: str, doc_type: str = "general") -> int:
    """
    Chunk and embed a document, then store in ChromaDB.
    Returns the number of chunks stored.
    """
    collection = _get_collection()
    embedder = _get_embedder()

    chunks = _chunk_text(content)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source, "doc_type": doc_type, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d chunks from source '%s'", len(chunks), source)
    return len(chunks)


# ─── Retrieve ─────────────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: Optional[int] = None, doc_type_filter: Optional[str] = None) -> dict:
    """
    Retrieve the most relevant chunks for a query.
    Returns {"documents": [...], "sources": [...]}
    """
    collection = _get_collection()
    embedder = _get_embedder()

    k = top_k or settings.rag_top_k
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]

    where_filter = {"doc_type": doc_type_filter} if doc_type_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, max(collection.count(), 1)),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents: List[str] = results["documents"][0] if results["documents"] else []
    metadatas: List[dict] = results["metadatas"][0] if results["metadatas"] else []
    sources = list({m.get("source", "unknown") for m in metadatas})

    return {"documents": documents, "sources": sources}


def get_collection_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


# ─── Seed knowledge base ──────────────────────────────────────────────────────

SEED_DOCUMENTS = [
    {
        "source": "behavioral_interview_guide",
        "doc_type": "interview_guide",
        "content": """
Behavioral Interview Questions and STAR Method Guide

The STAR method (Situation, Task, Action, Result) is the gold standard for answering behavioral questions.

Common Behavioral Questions:
1. Tell me about a time you dealt with a difficult coworker.
2. Describe a situation where you had to meet a tight deadline.
3. Give an example of a time you showed leadership.
4. Tell me about a time you failed and what you learned.
5. Describe a situation where you had to learn a new skill quickly.
6. Tell me about your greatest professional achievement.
7. Describe a time you resolved a conflict within your team.
8. Give an example of when you went above and beyond for a customer.

STAR Method Framework:
- Situation: Set the context. Describe the situation concisely.
- Task: Explain your responsibility in that situation.
- Action: Describe the specific steps YOU took.
- Result: Share the outcome, using metrics when possible.

Tips for behavioral answers:
- Keep answers to 2-3 minutes max
- Use 'I' not 'we' to highlight your contribution
- Always end with a positive result or learning
- Prepare 5-6 versatile STAR stories that can answer multiple questions
"""
    },
    {
        "source": "technical_interview_best_practices",
        "doc_type": "interview_guide",
        "content": """
Technical Interview Preparation Guide

System Design Questions:
- Design a URL shortener (like bit.ly)
- Design a social media feed
- Design a real-time chat application
- Design a distributed cache
- Design an e-commerce checkout system

Key areas to cover in system design:
1. Requirements clarification (functional vs non-functional)
2. Capacity estimation (traffic, storage, bandwidth)
3. High-level design (components, APIs)
4. Database design (SQL vs NoSQL, schema)
5. Scalability (load balancing, caching, sharding)
6. Fault tolerance and availability

Data Structures & Algorithms Topics:
- Arrays and Strings
- Linked Lists
- Trees and Graphs (BFS, DFS)
- Dynamic Programming
- Sorting and Searching
- Hash Tables
- Recursion and Backtracking
- Greedy Algorithms

Coding Interview Tips:
- Think aloud; interviewers value your reasoning process
- Start with brute force, then optimize
- Test your code with edge cases
- Time complexity: always mention Big-O
- Ask clarifying questions before coding
"""
    },
    {
        "source": "hr_interview_guidelines",
        "doc_type": "interview_guide",
        "content": """
HR Interview Questions and Best Practices

Common HR Questions:
1. Tell me about yourself. (Keep to 2 minutes, professional arc)
2. Why do you want to work here? (Research the company!)
3. Where do you see yourself in 5 years?
4. What are your greatest strengths and weaknesses?
5. Why are you leaving your current job?
6. What is your expected salary?
7. Do you have any questions for us?
8. Describe your ideal work environment.
9. How do you handle stress and pressure?
10. Are you willing to relocate or travel?

Answering "Tell me about yourself":
- Present → Past → Future structure
- 60-90 seconds
- Focus on professional highlights relevant to the role
- End with why you're excited about THIS role

Salary Negotiation Tips:
- Research market rates on LinkedIn, Glassdoor, Levels.fyi
- Give a range, not a specific number initially
- Consider total compensation (base + bonus + equity + benefits)
- Never accept on the spot; ask for time to consider

Questions to Ask the Interviewer:
- What does success look like in this role in the first 90 days?
- How would you describe the team culture?
- What are the biggest challenges facing the team right now?
- What opportunities are there for professional development?
"""
    },
    {
        "source": "software_engineering_roles",
        "doc_type": "job_description",
        "content": """
Software Engineering Role Requirements and Expectations

Junior Software Engineer (1-3 years):
- Proficiency in at least one programming language (Python, Java, JavaScript, Go)
- Understanding of data structures and algorithms
- Basic knowledge of version control (Git)
- Familiarity with REST APIs
- Ability to write unit tests
- Basic SQL knowledge

Mid-level Software Engineer (3-6 years):
- Strong proficiency in multiple languages/frameworks
- Experience with cloud platforms (AWS, Azure, GCP, IBM Cloud)
- Microservices architecture understanding
- CI/CD pipeline experience
- Code review and mentoring junior engineers
- Performance optimization experience
- System design fundamentals

Senior Software Engineer (6-10 years):
- Architectural decision-making
- Cross-team collaboration and technical leadership
- Distributed systems expertise
- Security best practices
- Scalability and reliability engineering
- Mentoring and hiring involvement
- Product sense and stakeholder communication

Principal / Staff Engineer (10+ years):
- Organization-wide technical strategy
- Drive engineering culture and standards
- Complex system decomposition
- External-facing technical communication
- Multi-year technical roadmap ownership
"""
    },
    {
        "source": "data_science_ml_roles",
        "doc_type": "job_description",
        "content": """
Data Science and Machine Learning Role Expectations

Data Scientist Interview Topics:
- Statistics: hypothesis testing, p-values, confidence intervals, distributions
- Machine Learning: supervised vs unsupervised, bias-variance tradeoff, regularization
- Algorithms: Linear/Logistic Regression, Decision Trees, Random Forest, XGBoost, Neural Networks
- Feature Engineering and selection
- Model evaluation: precision, recall, F1, ROC-AUC, RMSE
- SQL and data manipulation (pandas, NumPy)
- Data visualization (matplotlib, seaborn, Tableau)
- Experimentation: A/B testing design and analysis

ML Engineer Interview Topics:
- ML system design and deployment
- Model serving and inference optimization
- MLOps: MLflow, Kubeflow, SageMaker, Watson ML
- Data pipelines: Apache Spark, Airflow, Kafka
- Model monitoring and drift detection
- Containerization: Docker, Kubernetes
- Python engineering best practices
- Distributed training strategies

Common Data Science Questions:
1. Explain the bias-variance tradeoff.
2. How would you handle missing data?
3. What is regularization and when do you use L1 vs L2?
4. Explain the difference between bagging and boosting.
5. How do you deal with imbalanced datasets?
6. Explain how gradient descent works.
7. What is cross-validation and why is it important?
"""
    },
    {
        "source": "company_interview_processes",
        "doc_type": "company_info",
        "content": """
Industry Interview Processes at Top Tech Companies

Standard Interview Pipeline:
1. Resume Screen / ATS Filter
2. Recruiter Phone Screen (15-30 min, fit check)
3. Technical Phone Screen (45-60 min, coding/domain)
4. Onsite / Virtual Loop (4-6 rounds):
   - Coding rounds (2-3x)
   - System Design (1x, for mid+ level)
   - Behavioral / Leadership (1-2x)
   - Hiring Manager round
5. Reference Check
6. Offer

IBM Interview Process:
- Initial recruiter conversation
- Technical assessment or take-home
- Panel interview with team members
- Focus on IBM values: growth mindset, inclusivity, innovation
- Watson/watsonx AI product knowledge is a plus for AI roles
- Cloud (IBM Cloud) knowledge valued across roles

Preparation Timeline:
- 4+ weeks out: Review fundamentals, start LeetCode
- 3 weeks out: System design mock interviews
- 2 weeks out: Company research, STAR story prep
- 1 week out: Mock interviews, review past failures
- Day before: Light review, rest, confidence building

Post-Interview Follow-up:
- Send thank-you email within 24 hours
- Reference specific topics from the conversation
- Reiterate enthusiasm for the role
- Ask about timeline if not provided
"""
    },
    {
        "source": "soft_skills_leadership",
        "doc_type": "interview_guide",
        "content": """
Soft Skills and Leadership Assessment Guide

Communication Skills Assessment:
- Clarity: Can the candidate explain complex topics simply?
- Listening: Do they ask clarifying questions?
- Conciseness: Are answers focused and structured?
- Written communication: Portfolio, documentation quality

Leadership Indicators (even for non-manager roles):
- Taking ownership of problems without being asked
- Mentoring or helping teammates grow
- Driving projects to completion despite obstacles
- Influencing without authority
- Proactively communicating status and risks

Teamwork and Collaboration Questions:
1. How do you handle disagreements with teammates?
2. Tell me about a time you had to work with a difficult person.
3. Describe your experience working in agile/scrum teams.
4. How do you share knowledge with your team?

Problem-Solving and Critical Thinking:
- Structured thinking: MECE framework, first-principles
- Data-driven decision making
- Comfort with ambiguity
- Iterative, hypothesis-driven approach

Emotional Intelligence (EQ) Indicators:
- Self-awareness: Honest about weaknesses
- Empathy: Considers others' perspectives
- Resilience: Bounces back from setbacks
- Adaptability: Thrives in changing environments

Continuous Learning Mindset:
- Books, courses, certifications pursued
- Open-source contributions
- Side projects and experimentation
- Conference talks, blog posts
"""
    },
]


def seed_knowledge_base() -> int:
    """Seed ChromaDB with built-in interview preparation documents."""
    collection = _get_collection()
    if collection.count() > 0:
        logger.info("Knowledge base already seeded (%d docs). Skipping.", collection.count())
        return collection.count()

    total_chunks = 0
    for doc in SEED_DOCUMENTS:
        total_chunks += ingest_document(doc["content"], doc["source"], doc["doc_type"])

    logger.info("Knowledge base seeded with %d total chunks.", total_chunks)
    return total_chunks
