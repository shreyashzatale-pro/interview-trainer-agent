# Interview Trainer Agent — VS Code Setup

This is the VS Code-ready version of your notebook (`Interview_Trainer_Agent_RAG_Advanced.ipynb`).
All 9 notebook cells have been merged into one script: **`app.py`**.

## What changed vs. the notebook

1. **No `!pip install` cell.** Install dependencies once via `requirements.txt` (see below) instead of running pip inside the script.
2. **No hardcoded API key/project ID.** These are now read from a `.env` file using `python-dotenv`, so you never paste secrets into a script (or into this chat). If you shared a real key in the notebook before, treat it as compromised and regenerate it in IBM Cloud.
3. **`share=False` by default** in `app.launch(...)` at the bottom — better for local/VS Code use. Set it back to `True` if you specifically want a public Gradio tunnel link.
4. Everything else (knowledge base, RAG retrieval, resume parsing, scoring logic, the full glassmorphism/3D Gradio dashboard) is unchanged.

## Setup steps

1. **Open this folder in VS Code.**

2. **Create a virtual environment** (recommended), in the VS Code terminal:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your credentials:**
   - Copy `.env.example` to a new file named `.env` in the same folder.
   - Fill in your real `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` (and change `WATSONX_URL` if your Watson Studio project isn't in `eu-gb`).
   - `.env` is already listed in `.gitignore` so it won't get committed if you use git.

5. **Run it:**
   ```bash
   python app.py
   ```
   or press ▶️ "Run Python File" in VS Code with `app.py` open.

6. Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in your browser.

## Notes

- If your watsonx.ai project doesn't have `meta-llama/llama-3-3-70b-instruct` enabled, or you'd rather use a Granite chat model, change `GEN_MODEL_ID` near the top of `app.py`.
- The knowledge base (`KNOWLEDGE_BASE` list) is plain Python — extend it with more roles/questions any time.
- Resume upload requires a PDF; text is parsed with `pypdf`.
# interview-trainer-agent
