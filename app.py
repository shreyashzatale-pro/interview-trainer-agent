"""
Interview Trainer Agent — Advanced RAG Edition
Problem Statement No. 22 | AICTE 2026
Powered by IBM watsonx.ai (Granite embeddings + Llama generation) + RAG + Gradio)

This is a VS Code-ready conversion of the original Jupyter notebook.
Run with:  python app.py

Set your IBM Cloud credentials in a `.env` file (see .env.example) or as
environment variables before running — do NOT hardcode them in this file.
"""

import os
import re
import time

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load credentials from environment (.env file) instead of hardcoding them
# ---------------------------------------------------------------------------
load_dotenv()

WATSONX_APIKEY = os.getenv("YPSJiVAo6R8M0TAptuuOUgonFkN5TeLxdwU1C5iCLs4u")
WATSONX_URL = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("083d62de-9681-4ff0-975d-46b782dd2b13")

if not WATSONX_APIKEY or not WATSONX_PROJECT_ID:
    raise SystemExit(
        "\n❌ Missing IBM Cloud credentials.\n"
        "   Create a `.env` file next to this script (copy .env.example) and set:\n"
        "     WATSONX_APIKEY=...\n"
        "     WATSONX_PROJECT_ID=...\n"
        "     WATSONX_URL=https://eu-gb.ml.cloud.ibm.com   (or your region)\n"
    )

# ---------------------------------------------------------------------------
# Cell 2 — IBM watsonx.ai Setup
# ---------------------------------------------------------------------------
# NOTE: Not every watsonx.ai project has every model enabled — it depends on your
# plan and region. Run the check below FIRST if you're not sure what's available:
#
#   from ibm_watsonx_ai import APIClient
#   client = APIClient(credentials)
#   client.set.default_project(project_id)
#   print([m["model_id"] for m in client.foundation_models.get_model_specs()["resources"]])
#
# This project's available models do NOT include a Granite chat/instruct model
# (only Granite embedding + Granite time-series models are enabled here), so we
# satisfy the "IBM Granite mandatory" requirement via the embedding model that
# powers the RAG pipeline, and use Meta Llama (also an IBM Cloud Lite service via
# watsonx.ai) for text generation. If your project DOES have a Granite chat model
# (e.g. "ibm/granite-3-3-8b-instruct"), swap GEN_MODEL_ID below to use it instead.
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

credentials = {
    "apikey": WATSONX_APIKEY,
    "url": WATSONX_URL,
}
project_id = WATSONX_PROJECT_ID

GEN_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"          # text generation
EMBED_MODEL_ID = "ibm/granite-embedding-278m-multilingual"  # IBM Granite — powers the RAG retrieval

gen_params = {
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.TEMPERATURE: 0.7,
    GenParams.DECODING_METHOD: "sample",
    GenParams.TOP_P: 0.9,
    GenParams.REPETITION_PENALTY: 1.05,
}

model = ModelInference(
    model_id=GEN_MODEL_ID,
    credentials=credentials,
    project_id=project_id,
    params=gen_params,
)

embed_params = {EmbedParams.TRUNCATE_INPUT_TOKENS: 512}

embedder = Embeddings(
    model_id=EMBED_MODEL_ID,
    credentials=credentials,
    project_id=project_id,
    params=embed_params,
)

print("✅ Generation model + IBM Granite embedding model ready!")

# ---------------------------------------------------------------------------
# Cell 3 — RAG Knowledge Base
# ---------------------------------------------------------------------------
# A local corpus that stands in for "recruitment portals, professional networks,
# and company interview databases" — retrieved from via embeddings (RAG).
# Extend this list freely with more roles / questions / guidelines.

KNOWLEDGE_BASE = [
    # ---------------- Software Engineer ----------------
    {"role": "Software Engineer", "category": "technical", "difficulty": "Easy",
     "content": "Explain the difference between an array and a linked list, including time complexity of insertion and lookup."},
    {"role": "Software Engineer", "category": "technical", "difficulty": "Medium",
     "content": "Describe how you would design a rate limiter for a public API. Discuss token bucket vs sliding window."},
    {"role": "Software Engineer", "category": "technical", "difficulty": "Hard",
     "content": "Walk through designing a distributed key-value store with eventual consistency and partition tolerance."},
    {"role": "Software Engineer", "category": "behavioral", "difficulty": "Medium",
     "content": "Tell me about a time you disagreed with a teammate's technical decision. How did you resolve it?"},
    {"role": "Software Engineer", "category": "hr_guideline", "difficulty": "Any",
     "content": "Top tech companies weight code readability and testing habits as heavily as raw problem-solving speed."},

    # ---------------- Data Scientist ----------------
    {"role": "Data Scientist", "category": "technical", "difficulty": "Easy",
     "content": "What is the bias-variance tradeoff and how does it affect model selection?"},
    {"role": "Data Scientist", "category": "technical", "difficulty": "Medium",
     "content": "How would you handle a highly imbalanced classification dataset in a fraud detection system?"},
    {"role": "Data Scientist", "category": "technical", "difficulty": "Hard",
     "content": "Design an end-to-end ML pipeline for real-time recommendation, covering feature store, training, and drift monitoring."},
    {"role": "Data Scientist", "category": "behavioral", "difficulty": "Medium",
     "content": "Describe a project where your model's results contradicted stakeholder expectations. How did you communicate it?"},
    {"role": "Data Scientist", "category": "hr_guideline", "difficulty": "Any",
     "content": "Interviewers for data roles often probe both statistical rigor and the ability to explain results to non-technical audiences."},

    # ---------------- Product Manager ----------------
    {"role": "Product Manager", "category": "technical", "difficulty": "Medium",
     "content": "How would you prioritize a backlog when engineering capacity is cut by 30% next quarter?"},
    {"role": "Product Manager", "category": "technical", "difficulty": "Hard",
     "content": "Design a metrics framework (North Star + guardrail metrics) for a new social-sharing feature."},
    {"role": "Product Manager", "category": "behavioral", "difficulty": "Medium",
     "content": "Tell me about a time you had to say no to a senior stakeholder's feature request. What happened?"},
    {"role": "Product Manager", "category": "hr_guideline", "difficulty": "Any",
     "content": "PM interviews typically blend product sense, execution, analytical, and leadership/behavioral rounds."},

    # ---------------- Frontend Developer ----------------
    {"role": "Frontend Developer", "category": "technical", "difficulty": "Easy",
     "content": "Explain the virtual DOM and why it improves rendering performance in frameworks like React."},
    {"role": "Frontend Developer", "category": "technical", "difficulty": "Medium",
     "content": "How would you optimize a web app's Largest Contentful Paint (LCP) score?"},
    {"role": "Frontend Developer", "category": "technical", "difficulty": "Hard",
     "content": "Design a component library that supports theming, accessibility, and tree-shaking across multiple products."},
    {"role": "Frontend Developer", "category": "behavioral", "difficulty": "Medium",
     "content": "Describe a time you pushed back on a design that was hard to implement accessibly."},

    # ---------------- Backend Developer ----------------
    {"role": "Backend Developer", "category": "technical", "difficulty": "Easy",
     "content": "What is the difference between SQL and NoSQL databases, and when would you choose each?"},
    {"role": "Backend Developer", "category": "technical", "difficulty": "Medium",
     "content": "How would you design an idempotent payment processing API?"},
    {"role": "Backend Developer", "category": "technical", "difficulty": "Hard",
     "content": "Design a message queue based microservice architecture that guarantees at-least-once delivery."},
    {"role": "Backend Developer", "category": "behavioral", "difficulty": "Medium",
     "content": "Tell me about a production incident you handled. What was the root cause and the fix?"},

    # ---------------- Generic HR / Behavioral (applies to all roles) ----------------
    {"role": "Any", "category": "behavioral", "difficulty": "Easy",
     "content": "Tell me about yourself and why you're interested in this role."},
    {"role": "Any", "category": "behavioral", "difficulty": "Medium",
     "content": "Describe a situation where you had to work under a tight deadline with limited resources."},
    {"role": "Any", "category": "behavioral", "difficulty": "Medium",
     "content": "Give an example of receiving critical feedback. How did you respond and what changed afterward?"},
    {"role": "Any", "category": "behavioral", "difficulty": "Hard",
     "content": "Tell me about the biggest failure in your career and what you learned from it."},
    {"role": "Any", "category": "hr_guideline", "difficulty": "Any",
     "content": "The STAR method (Situation, Task, Action, Result) is the standard structure recruiters expect for behavioral answers."},
    {"role": "Any", "category": "hr_guideline", "difficulty": "Any",
     "content": "Candidates are expected to research the company's mission, recent news, and products before the interview."},
    {"role": "Any", "category": "hr_guideline", "difficulty": "Any",
     "content": "Confident body language, structured answers, and concrete metrics (numbers/impact) significantly raise HR scoring."},
]

print(f"✅ Knowledge base loaded with {len(KNOWLEDGE_BASE)} entries across "
      f"{len(set(k['role'] for k in KNOWLEDGE_BASE))} roles.")

# ---------------------------------------------------------------------------
# Cell 4 — RAG Retrieval Engine (embeddings + cosine similarity)
# ---------------------------------------------------------------------------
import numpy as np

def _embed_texts(texts):
    result = embedder.embed_documents(texts=texts)
    return np.array(result)

print("⏳ Building vector index for the knowledge base (one-time)...")
_kb_texts = [item["content"] for item in KNOWLEDGE_BASE]
KB_VECTORS = _embed_texts(_kb_texts)
print("✅ Vector index built:", KB_VECTORS.shape)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return a @ b

def retrieve_context(query, role, top_k=4):
    """Retrieve the most relevant knowledge-base snippets for a query,
    preferring entries tagged with the candidate's role (falling back to 'Any')."""
    query_vec = np.array(embedder.embed_query(text=query))
    sims = cosine_sim(KB_VECTORS, query_vec)

    scored = []
    for idx, item in enumerate(KNOWLEDGE_BASE):
        boost = 0.15 if item["role"] == role else (0.05 if item["role"] == "Any" else 0.0)
        scored.append((sims[idx] + boost, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    context_block = "\n".join(f"- [{i['category'].upper()}] {i['content']}" for _, i in top)
    return context_block, [i for _, i in top]

print("✅ RAG retrieval engine ready!")

# ---------------------------------------------------------------------------
# Cell 5 — Resume / Job Title Parsing
# ---------------------------------------------------------------------------
from pypdf import PdfReader

def extract_resume_text(file_path):
    """Extracts raw text from an uploaded resume PDF. Returns '' if no file."""
    if not file_path:
        return ""
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()[:4000]   # cap length sent to the LLM
    except Exception as e:
        return f"[Could not read resume: {e}]"

print("✅ Resume parser ready!")

# ---------------------------------------------------------------------------
# Cell 6 — Core Agent Functions (RAG-augmented, IBM Granite powered)
# ---------------------------------------------------------------------------

import time, re

def safe_generate(prompt):
    try:
        return model.generate_text(prompt).strip()
    except Exception as e:
        return f"[Model error: {e}]"

def build_candidate_profile(name, experience, role, job_title_or_resume_text):
    return {
        "name": name or "Candidate",
        "experience": experience,
        "role": role,
        "extra_context": (job_title_or_resume_text or "")[:2000],
    }

def generate_prep_strategy(profile):
    context, _ = retrieve_context(f"{profile['role']} interview preparation guidelines", profile["role"], top_k=5)
    prompt = f"""You are a career coach preparing {profile['name']} ({profile['experience']} level)
for a {profile['role']} interview.

Relevant guidelines retrieved from HR/company databases:
{context}

Candidate background / resume notes:
{profile['extra_context'] or 'Not provided'}

Write a short, punchy preparation strategy (max 6 bullet points) covering what to
revise, how to structure behavioral answers, and 1 confidence tip.
Start each bullet with one relevant emoji (e.g. 📚 🎯 💬 🧠 ⏱️ 💪)."""
    return safe_generate(prompt)

def ask_question(profile, difficulty, q_type):
    query = f"{profile['role']} {q_type} {difficulty} interview question"
    context, sources = retrieve_context(query, profile["role"], top_k=4)
    prompt = f"""You are a strict technical interviewer at a top company, interviewing
{profile['name']} for a {profile['role']} role ({profile['experience']} level).

Use the retrieved reference material below as inspiration (do not copy verbatim):
{context}

Ask ONE {difficulty}-level {q_type} interview question about {profile['role']}.
Only output the question itself. No preamble, no labels, no explanation. End with '?'."""
    raw = safe_generate(prompt)
    clean = raw.replace("```", "").replace("Question:", "").strip()
    parts = clean.split("?")
    question = (parts[0].strip() + "?") if parts and parts[0].strip() else clean
    return question, sources

def generate_model_answer(profile, question, difficulty):
    prompt = f"""You are a Senior {profile['role']} giving a model interview answer.
Question: {question}
Difficulty: {difficulty}

Provide a concise, well-structured ideal answer (use STAR format if it's a
behavioral question). Max 120 words."""
    return safe_generate(prompt)

def evaluate_answer(profile, question, answer, time_taken):
    if not answer or not answer.strip():
        return None
    prompt = f"""You are a Senior Technical + HR Interviewer evaluating a candidate.
Role: {profile['role']} | Experience: {profile['experience']} | Time Taken: {time_taken}s
Question: {question}
Candidate Answer: {answer}

Respond in EXACTLY this format:
TECHNICAL_ACCURACY: X/10
COMMUNICATION: X/10
PROBLEM_SOLVING: X/10
CONFIDENCE: X/10
MISSING: [2-3 missing concepts, comma separated]
STRENGTHS: [1-2 strengths, comma separated]
FEEDBACK: [2-3 lines of constructive feedback]
VERDICT: PASS or FAIL
NEXT_QUESTION_HINT: [what topic to study next]"""
    return safe_generate(prompt)

def parse_evaluation(raw):
    result = {"technical": 5, "communication": 5, "problem_solving": 5, "confidence": 5,
              "missing": "N/A", "strengths": "N/A", "feedback": "N/A",
              "verdict": "FAIL", "next_hint": "N/A"}
    if not raw:
        return result
    for line in raw.split("\n"):
        line = line.strip()
        try:
            if line.startswith("TECHNICAL_ACCURACY:"):
                result["technical"] = int(re.search(r"\d+", line).group())
            elif line.startswith("COMMUNICATION:"):
                result["communication"] = int(re.search(r"\d+", line).group())
            elif line.startswith("PROBLEM_SOLVING:"):
                result["problem_solving"] = int(re.search(r"\d+", line).group())
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = int(re.search(r"\d+", line).group())
            elif line.startswith("MISSING:"):
                result["missing"] = line.replace("MISSING:", "").strip()
            elif line.startswith("STRENGTHS:"):
                result["strengths"] = line.replace("STRENGTHS:", "").strip()
            elif line.startswith("FEEDBACK:"):
                result["feedback"] = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("VERDICT:"):
                result["verdict"] = "PASS" if "PASS" in line.upper() else "FAIL"
            elif line.startswith("NEXT_QUESTION_HINT:"):
                result["next_hint"] = line.replace("NEXT_QUESTION_HINT:", "").strip()
        except Exception:
            pass
    return result

def generate_tips(role):
    prompt = (f"Give 3 quick interview tips for a {role} role. Each tip under 10 words. "
              f"Bullet points only, each starting with one relevant emoji.")
    return safe_generate(prompt)

print("✅ Core RAG-powered agent functions ready!")

# ---------------------------------------------------------------------------
# Cell 7 — Hi-Tech Gradio Dashboard (glassmorphism + animated 3D radar/cube)
# ---------------------------------------------------------------------------

import gradio as gr
import plotly.graph_objects as go

session = {
    "profile": None, "question": "", "start_time": 0,
    "score_history": [], "metric_history": [],
    "question_count": 0, "pass_count": 0
}

CSS = """
.gradio-container {
    background: radial-gradient(circle at 20% 20%, #1a1a3d 0%, #05050f 60%) !important;
    font-family: 'Segoe UI', sans-serif !important;
}
#header-box {
    position: relative;
    background: linear-gradient(120deg, #00c6ff, #7b2ff7, #ff00c8);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border-radius: 18px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 0 40px rgba(123, 47, 247, 0.55);
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
}
.about-tab * {
    color: White !important;
} 
textarea{
    color:Black !important;
}

input[type="text"]{
    color:black!important;
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
/* --- Hi-tech rotating 3D cube (pure CSS 3D transform) --- */
.cube-stage {
    width: 70px; height: 70px;
    perspective: 500px;
    flex-shrink: 0;
}
.cube {
    width: 100%; height: 100%;
    position: relative;
    transform-style: preserve-3d;
    animation: spin3d 9s infinite linear;
}
.cube div {
    position: absolute;
    width: 70px; height: 70px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 10px;
    backdrop-filter: blur(4px);
}
.cube .f1 { transform: translateZ(35px); }
.cube .f2 { transform: rotateY(90deg) translateZ(35px); }
.cube .f3 { transform: rotateY(180deg) translateZ(35px); }
.cube .f4 { transform: rotateY(-90deg) translateZ(35px); }
.cube .f5 { transform: rotateX(90deg) translateZ(35px); }
.cube .f6 { transform: rotateX(-90deg) translateZ(35px); }
@keyframes spin3d {
    from { transform: rotateX(0deg) rotateY(0deg); }
    to   { transform: rotateX(360deg) rotateY(360deg); }
}
.glass {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
}
.glass{
    padding:18px;
}
button.primary {
    background: linear-gradient(90deg, #00c6ff, #7b2ff7) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(0, 198, 255, 0.5);
    transition: transform 0.15s ease !important;
}
button.primary:hover { transform: translateY(-2px) scale(1.02); }
button.secondary {
    background: linear-gradient(90deg, #11998e, #38ef7d) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #05050f !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(56, 239, 125, 0.45);
    transition: transform 0.15s ease !important;
}
button.secondary:hover { transform: translateY(-2px) scale(1.02); }
"""

def make_radar_chart(ev):
    categories = ["🔬 Technical", "💬 Communication", "🧠 Problem Solving", "💪 Confidence"]
    values = [ev["technical"], ev["communication"], ev["problem_solving"], ev["confidence"]]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line=dict(color="#00c6ff", width=3),
        fillcolor="rgba(123, 47, 247, 0.35)",
        marker=dict(size=8, color="#ff00c8")
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], color="white", gridcolor="rgba(255,255,255,0.2)"),
            angularaxis=dict(color="white", gridcolor="rgba(255,255,255,0.2)")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=360,
        title=dict(text="📊 Skill Radar", font=dict(color="white"))
    )
    return fig

def make_progress_chart():
    hist = session["score_history"]
    fig = go.Figure()
    if hist:
        fig.add_trace(go.Scatter(
            y=hist, mode="lines+markers",
            line=dict(color="#38ef7d", width=3),
            marker=dict(size=9, color="#00c6ff")
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(range=[0, 10], title="Avg Score", gridcolor="rgba(255,255,255,0.15)"),
        xaxis=dict(title="Question #", gridcolor="rgba(255,255,255,0.15)"),
        margin=dict(l=40, r=20, t=30, b=40), height=300,
        title=dict(text="📈 Score Trend", font=dict(color="white"))
    )
    return fig

def make_3d_score_cube():
    """A genuine interactive 3D visualization: each metric's trajectory
    across every question answered so far, plotted in 3D space
    (x = question #, y = metric, z = score). Drag to rotate!"""
    hist = session["metric_history"]
    fig = go.Figure()
    metric_names = ["Technical", "Communication", "Problem Solving", "Confidence"]
    colors = ["#00c6ff", "#38ef7d", "#ff00c8", "#ffd166"]
    emojis = ["🔬", "💬", "🧠", "💪"]

    if hist:
        qnums = list(range(1, len(hist) + 1))
        for m_idx, m_name in enumerate(metric_names):
            key = m_name.lower().replace(" ", "_")
            z_vals = [h[key] for h in hist]
            fig.add_trace(go.Scatter3d(
                x=qnums, y=[m_idx] * len(qnums), z=z_vals,
                mode="lines+markers",
                name=f"{emojis[m_idx]} {m_name}",
                line=dict(color=colors[m_idx], width=5),
                marker=dict(size=5, color=colors[m_idx])
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Question #", color="white", gridcolor="rgba(255,255,255,0.2)",
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Metric", color="white", gridcolor="rgba(255,255,255,0.2)",
                       backgroundcolor="rgba(0,0,0,0)",
                       tickvals=[0, 1, 2, 3], ticktext=metric_names),
            zaxis=dict(title="Score /10", range=[0, 10], color="white",
                       gridcolor="rgba(255,255,255,0.2)", backgroundcolor="rgba(0,0,0,0)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
        title=dict(text="🧊 3D Skill Journey — rotate & explore!", font=dict(color="white"))
    )
    return fig

def setup_profile(name, experience, role, job_title, resume_file):
    resume_text = extract_resume_text(resume_file.name if resume_file else None)
    extra_context = (job_title or "") + ("\n" + resume_text if resume_text else "")
    profile = build_candidate_profile(name, experience, role, extra_context)
    session["profile"] = profile
    session["question_count"] = 0
    session["pass_count"] = 0
    session["score_history"] = []
    session["metric_history"] = []
    strategy = generate_prep_strategy(profile)
    status = (f"### 🎉 Profile ready for **{profile['name']}** — 🎓 {role} | ⚡ {experience}\n"
              f"🚀 Head to the **🎯 Interview** tab to begin!")
    return status, strategy

def start_interview(difficulty, q_type):
    if not session["profile"]:
        return "", "⚠️ Set up your profile in the **👤 Profile & Resume** tab first!", "", None
    profile = session["profile"]
    session["start_time"] = time.time()
    session["question_count"] += 1
    question, sources = ask_question(profile, difficulty, q_type)
    session["question"] = question
    tips = generate_tips(profile["role"])
    used = "\n".join(f"🔹 [{s['category']}] {s['content'][:70]}..." for s in sources[:3])
    status = f"""🎯 **Question #{session['question_count']}** 🔥
📚 Role: {profile['role']} | ⚡ {difficulty} | 🎭 {q_type}
✅ Attempted: {session['question_count']} | 🏆 Passed: {session['pass_count']}

**🔎 RAG sources used:**
{used}"""
    return question, status, tips, None

def submit_answer(answer, difficulty):
    if not session["question"]:
        return "", "⚠️ Start an interview first!", "", "", "", None, "", None
    if not answer or not answer.strip():
        return "", "⚠️ Please type an answer!", "", "", "", None, "", None

    profile = session["profile"]
    time_taken = int(time.time() - session["start_time"])
    raw = evaluate_answer(profile, session["question"], answer, time_taken)
    ev = parse_evaluation(raw)
    avg = (ev["technical"] + ev["communication"] + ev["problem_solving"] + ev["confidence"]) / 4
    session["score_history"].append(avg)
    session["metric_history"].append({
        "technical": ev["technical"], "communication": ev["communication"],
        "problem_solving": ev["problem_solving"], "confidence": ev["confidence"]
    })
    if ev["verdict"] == "PASS":
        session["pass_count"] += 1

    model_answer = generate_model_answer(profile, session["question"], difficulty)

    verdict = "✅ PASS 🎉" if ev["verdict"] == "PASS" else "❌ FAIL 💪 (keep going!)"
    missing = f"❌ Missing:\n{ev['missing']}"
    strengths = f"✅ Strengths:\n{ev['strengths']}"
    feedback = f"💡 Feedback:\n{ev['feedback']}\n\n📖 Study Next:\n{ev['next_hint']}"
    radar = make_radar_chart(ev)
    n = len(session["score_history"])
    overall = sum(session["score_history"]) / n if n else 0
    stats = f"""📈 **Session Stats** 🏆
✅ Total: {session['question_count']} | 🏆 Passed: {session['pass_count']} | ❌ Failed: {session['question_count'] - session['pass_count']}
⭐ Session Average: {overall:.1f}/10"""

    return verdict, missing, strengths, feedback, stats, radar, model_answer, make_3d_score_cube()

with gr.Blocks(title="🤖 Interview Trainer Agent — RAG Edition") as app:

    gr.HTML("""
    <div id="header-box">
        <div class="cube-stage">
            <div class="cube">
                <div class="f1">🎯</div>
                <div class="f2">💻</div>
                <div class="f3">📊</div>
                <div class="f4">🧠</div>
                <div class="f5">✅</div>
                <div class="f6">🏆</div>
            </div>
        </div>
        <div>
            <h1 style="color:white; margin:0; font-size:2.3em; text-shadow:0 0 12px rgba(0,0,0,0.4);">
                🤖 Interview Trainer Agent
            </h1>
            <p style="color:rgba(255,255,255,0.9); margin:6px 0 0 0; font-size:1.05em;">
                🔮 RAG-Powered · 💠 IBM watsonx.ai Granite · 🎓 AICTE 2026 · 📌 Problem Statement No. 22
            </p>
        </div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("👤 Profile & Resume"):
            with gr.Row():
                with gr.Column(elem_classes="glass"):
                    name_in = gr.Textbox(label="🧑 Full Name")
                    exp_in = gr.Dropdown(
                        choices=["Fresher / 0-1 yr", "Junior / 1-3 yrs", "Mid / 3-6 yrs", "Senior / 6+ yrs"],
                        label="⚡ Experience Level", value="Junior / 1-3 yrs"
                    )
                    role_in = gr.Dropdown(
                        choices=["Software Engineer", "Data Scientist", "Product Manager",
                                 "Frontend Developer", "Backend Developer"],
                        label="🎓 Target Job Role", value="Software Engineer"
                    )
                    job_title_in = gr.Textbox(
                        label="📄 Job Title / Job Description (optional, paste text)", lines=3
                    )
                    resume_in = gr.File(label="📎 Upload Resume (PDF, optional)", file_types=[".pdf"])
                    setup_btn = gr.Button("🚀 Build Profile & Strategy", variant="primary", size="lg")
                with gr.Column(elem_classes="glass"):
                    profile_status = gr.Markdown("### 👆 Fill your profile and click Build")
                    strategy_box = gr.Textbox(label="🧭 AI Prep Strategy (RAG-generated)", lines=10, interactive=False)

        with gr.Tab("🎯 Interview"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass"):
                    difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="⚡ Difficulty", value="Medium")
                    q_type = gr.Dropdown(
                        choices=["technical", "behavioral", "hr_guideline"],
                        label="🎭 Question Type", value="technical"
                    )
                    start_btn = gr.Button("🚀 Get Question", variant="primary", size="lg")
                    tips_box = gr.Textbox(label="💡 Quick Tips", lines=4, interactive=False)
                with gr.Column(scale=2, elem_classes="glass"):
                    status_box = gr.Markdown("### 👆 Click 'Get Question' to begin")
                    question_box = gr.Textbox(label="📝 Question", lines=3, interactive=False)
                    answer_box = gr.Textbox(label="✍️ Your Answer", lines=6, placeholder="Type your answer here...")
                    submit_btn = gr.Button("✅ Submit Answer", variant="secondary", size="lg")

        with gr.Tab("📊 Results"):
            with gr.Row():
                verdict_box = gr.Textbox(label="🏆 Verdict", lines=2, interactive=False)
                radar_plot = gr.Plot(label="📊 Skill Radar")
            with gr.Row():
                strengths_box = gr.Textbox(label="✅ Strengths", lines=3, interactive=False)
                missing_box = gr.Textbox(label="❌ Improve", lines=3, interactive=False)
            feedback_box = gr.Textbox(label="💡 Feedback & Study Tips", lines=4, interactive=False)
            model_answer_box = gr.Textbox(label="🌟 Model Answer (AI-generated)", lines=5, interactive=False)

        with gr.Tab("📈 Progress"):
            stats_box = gr.Markdown("### 📊 Answer questions to see your stats!")
            progress_plot = gr.Plot(label="📈 Score Trend")
            cube_plot = gr.Plot(label="🧊 3D Skill Journey")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🤖 Interview Trainer Agent — RAG Edition

            ### 🎯 Features
            - 👤 Profile-based tailoring: name, experience level, job role, resume/job title
            - 🔎 **RAG retrieval** from a role-tagged knowledge base (questions, behavioral scenarios, HR guidelines)
            - 🤖 AI-generated **questions, model answers, and improvement tips**
            - 📊 Combined **technical + soft-skill** scoring (4 metrics)
            - 🕸️ Animated skill radar + 🧊 rotatable **3D skill-journey chart** + 📈 progress trend

            ### 🛠️ Tech Stack
            | Component | Technology |
            |-----------|------------|
            | 🧠 LLM | **IBM Granite** (watsonx.ai) |
            | 🔎 Embeddings / RAG | IBM Slate embeddings + cosine similarity |
            | 🎨 UI | Gradio + Plotly (incl. 3D) |
            | 📄 Resume Parsing | pypdf |
            | 🐍 Language | Python 3.12 |

            **🎓 Built for AICTE 2026 | 📌 Problem Statement No. 22**
            """)

    setup_btn.click(fn=setup_profile,
                     inputs=[name_in, exp_in, role_in, job_title_in, resume_in],
                     outputs=[profile_status, strategy_box])

    start_btn.click(fn=start_interview,
                     inputs=[difficulty, q_type],
                     outputs=[question_box, status_box, tips_box, radar_plot])

    submit_btn.click(fn=submit_answer,
                      inputs=[answer_box, difficulty],
                      outputs=[verdict_box, missing_box, strengths_box, feedback_box,
                               stats_box, radar_plot, model_answer_box, cube_plot])

print("✅ Hi-tech 3D dashboard built!")