📚 AI Document RAG Assistant
A Streamlit app that turns PDFs into a chat-enabled Retrieval-Augmented Generation (RAG) assistant. Upload resumes, job descriptions, or any PDF and ask natural-language questions with context-aware answers powered by Perplexity (or your LLM), FAISS vector search, and HuggingFace embeddings.
🔥 Key Features
Upload any PDF and instantly create an indexed, embeddable vector store.
Chat interface that answers questions using only the document context (no hallucination).
Pre-built tab for a default resume (e.g., resume.pdf) for quick demo.
JD-to-resume matching, optimized resume generation, and tailored STAR interview answers.
Configurable chunk size, overlap, and retrieval K.
Clean Streamlit UI with metrics, source chunk traceability, and download/copy-ready outputs.
🎯 Repo Contents
.
├─ app.py                 # Main Streamlit app (the code you provided)
├─ resume.pdf             # default demo resume (used in 'Dhanesh' tab)
├─ requirements.txt       # Python deps
├─ Dockerfile             # optional: dockerize app
├─ .github/workflows/ci.yml  # optional: CI for linting/tests/deploy
└─ README.md
🚀 Quick Start (Local)
Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
Create a virtual environment and install dependencies (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
Set environment variables (example for UNIX shells)
export PERPLEXITY_API_KEY="your_real_perplexity_api_key_here"
# Optional: configure other keys if you add extra LLM/backends later
Run the app
streamlit run app.py
Visit http://localhost:8501 and explore the tabs.
⚙️ Environment Configuration
Create a .env (or add variables to your system) with:
PERPLEXITY_API_KEY=your_perplexity_key_here
# Optional:
# HF_TOKEN=your_huggingface_token
# OTHER_API_KEY=...
🔐 Security note: Never commit .env or API keys to the repository. Add .env to .gitignore.
🧱 Architecture Overview
Frontend: Streamlit for interactive UI and chat display.
Retrieval: FAISS vector index from LangChain community vectorstore.
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (via HuggingFaceEmbeddings).
LLM: Calls to Perplexity API (sonar-pro in your code), used for summarization, JD analysis, and answer generation.
RAG Flow:
Upload PDF → split into chunks → embed → store in FAISS.
On user query → similarity_search to get top chunks → LLM prompt composed with context → answer returned.
State Management: st.session_state tracks vectorstores, messages, and job-specific outputs.
🛠️ Configuration Flags in Code
At the top of app.py you’ll find tunable constants:
MODEL = "sonar-pro"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K_DOCUMENTS = 10
MAX_HISTORY_MESSAGES = 6
Adjust CHUNK_SIZE, CHUNK_OVERLAP, and TOP_K_DOCUMENTS to trade off recall vs. context size for the LLM.
🔍 How to Use
Tab 1 — Personalized Document:
Uses the local resume.pdf loaded at startup to answer questions about that resume.
Tab 2 — General Document Uploader:
Upload any PDF — the app will index it, auto-generate a short summary, then allow chat queries against it.
Tab 3 — JD Match Analyzer / Generator:
Upload or select a resume (default is the built-in one).
Paste job description text into the JD area.
Click:
Analyze JD Match — returns match score, matching skills, gaps & suggestions.
Generate Optimized Resume — produces an ATS-optimized markdown resume draft.
Generate Interview Prep — generates three STAR responses tailored to the JD.
📸 Screenshots / Demo
Insert screenshots or short GIFs here to show the UI.
Example:
🐳 Docker (optional)
Use this Dockerfile snippet to containerize:
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
Build & run:
docker build -t rag-assistant .
docker run -e PERPLEXITY_API_KEY="$PERPLEXITY_API_KEY" -p 8501:8501 rag-assistant
✅ Deployment Recommendations
Streamlit Cloud: Easiest if you want a quick public demo (remember to configure secrets in the Streamlit Cloud UI).
Heroku / Render / Fly / Railway: Use Docker or pip install, set environment variables, and expose port 8501.
Security: Use platform secrets for API keys (do not commit keys).
🔒 Security & Privacy Notes
The app sends document snippets to the LLM provider (Perplexity). Don’t upload sensitive/confidential PDFs unless you trust your provider and compliance needs are satisfied.
Mask or redact PII before uploading if necessary.
Limit access to your deployment (auth) for production use.
🧪 Testing & Troubleshooting
Common problems
PERPLEXITY_API_KEY not set → ensure env var is defined and app restarted.
Failed to process PDF → validate the PDF; try opening locally; some PDFs with heavy images or DRM may fail.
Slow indexing → reduce CHUNK_SIZE or use smaller PDFs for testing.
Helpful commands
pip install -r requirements.txt
streamlit cache clear       # Clear cached resources (vectorstore caches)
♻️ Contribution
Contributions are welcome! Suggested ways to contribute:
Improve the UI/UX (add login, file management).
Add support for alternative LLMs (OpenAI, Anthropic) via config.
Add tests for key functions.
Improve prompt design for better, less hallucination-prone answers.
Please open issues or PRs with clear descriptions and rationale.
🧾 Example requirements.txt
streamlit>=1.24
requests>=2.28
langchain-community>=0.0.XX
langchain-text-splitters>=0.0.XX
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
(Adjust versions to match your environment and resolve compatibility.)
📌 License
MIT License — see LICENSE for details.
🙏 Credits & Acknowledgements
Built using Streamlit, LangChain community components, FAISS, and SentenceTransformers.
Prompt engineering inspired by practical RAG patterns and ATS resume optimization techniques.
Sample README badges & links to add (optional)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/<your-username>/<your-repo>/main/app.py)
