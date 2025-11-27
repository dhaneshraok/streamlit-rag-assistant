import streamlit as st
import requests
import os
import time
import tempfile 
import json 
from io import BytesIO
import os
import re # Added for score parsing

# ==========================
# CONFIG & API Setup
# ==========================

# 🛑 ACTION REQUIRED: REPLACE THIS PLACEHOLDER WITH YOUR ACTUAL PERPLEXITY API KEY
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "YOUR_PERPLEXITY_API_KEY_NOT_SET")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K_DOCUMENTS = 10 
MAX_HISTORY_MESSAGES = 6 

# --- Keys for State Management ---
DHANESH_VECTORSTORE_KEY = "dhanesh_vectorstore"
DHANESH_MESSAGES_KEY = "dhanesh_messages"
DHANESH_FILE_PATH = "resume.pdf" 
GENERAL_VECTORSTORE_KEY = "general_vectorstore"
GENERAL_MESSAGES_KEY = "general_messages"
GENERAL_FILE_HASH_KEY = "general_file_hash"
PROCESSING_STATE_KEY = "is_processing" 
GENERAL_SUMMARY_GENERATED = "general_summary_done" 
# JD Match State
JD_TEXT_KEY = "jd_text"
JD_MATCH_RESULT_KEY = "jd_match_result"
JD_UPLOADED_VECTORSTORE_KEY = "jd_uploaded_vectorstore"
JD_UPLOADED_FILE_NAME = "jd_uploaded_file_name"
OPTIMIZED_RESUME_RESULT_KEY = "optimized_resume_result"
INTERVIEW_PREP_RESULT_KEY = "interview_prep_result"

# --- Custom Styling Variables ---
PRIMARY_ACCENT = "#0088ff"  # Bright Blue/Green for highlights
BACKGROUND_DARK = "#1a1a1a" # Main content Deep Charcoal/Black
BACKGROUND_MEDIUM = "#f0f2f6" # Light Sidebar Background
TEXT_LIGHT = "#000000"      # Dark text for light sidebar
TEXT_DARK = "#ffffff"       # White text for dark main area

# ==========================
# DEPENDENCIES & HELPERS
# ==========================

def load_langchain_dependencies():
    """Loads and returns the necessary LangChain components."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    # RecursiveCharacterTextSplitter is the corrected class name
    return PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, FAISS

def ask_perplexity(full_prompt):
    """Sends the complete, context-aware prompt to the Perplexity API."""
    if PERPLEXITY_API_KEY == "YOUR_PERPLEXITY_API_KEY_HERE" or not PERPLEXITY_API_KEY:
         return "Error: PERPLEXITY_API_KEY not set. Please update the placeholder in the code."
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }
    try:
        response = requests.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60) 
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_data = response.json()
                error_text = error_data.get("error", {}).get("message", response.text)
            except json.JSONDecodeError:
                pass

            if response.status_code == 401:
                return f"Error: Perplexity API returned status code 401 (Unauthorized). Please check your **PERPLEXITY_API_KEY**."
            else:
                 return f"Error: Perplexity API returned status code {response.status_code}. Response: {error_text[:200]}..."
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: Failed during LLM call. Details: {e}"

@st.cache_resource(hash_funcs={str: lambda x: x})
def get_vectorstore(file_like_object):
    """Loads, splits, and builds FAISS vector store from a given file-like object (or path)."""
    PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, FAISS = load_langchain_dependencies()
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            if isinstance(file_like_object, str):
                with open(file_like_object, 'rb') as f_in:
                    tmp_file.write(f_in.read())
            elif hasattr(file_like_object, 'read'):
                file_like_object.seek(0) # Ensure we read from the start
                tmp_file.write(file_like_object.read())
            
            temp_file_path = tmp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading the document: {e}. Please ensure the file is valid and accessible.")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "•", "|", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def get_dhanesh_vectorstore():
    """Loads the hardcoded resume.pdf file once at startup."""
    if not os.path.exists(DHANESH_FILE_PATH):
        st.error(f"❌ **File Not Found:** The file `'{DHANESH_FILE_PATH}'` was not found. Please place your resume/document in the project directory.")
        return None
    
    return get_vectorstore(DHANESH_FILE_PATH)

def generate_document_summary(vectorstore, doc_name):
    """
    Generates a summary and identifies the document type for the general RAG tab.
    """
    
    top_chunks = vectorstore.similarity_search("", k=30) 
    context = "\n\n".join([doc.page_content for doc in top_chunks])

    summary_prompt = f"""
    Analyze the following document snippets provided as context. Your task is to:
    1.  Determine the **Document Type** (e.g., Technical Whitepaper, Financial Report, Academic Paper, Resume/CV, Legal Contract, Product Manual).
    2.  Provide a **Concise Summary** (3-4 sentences) outlining the main topics and purpose of the document.

    Format your response *strictly* as follows:
    **Document Type:** [Your determined type]
    **Summary:** [Your 3-4 sentence summary]

    ---
    Context (Snippets from "{doc_name}"):
    {context}
    ---
    """
    
    return ask_perplexity(summary_prompt)

def analyze_jd_match(jd_text, resume_vectorstore, resume_name):
    """
    Analyzes the resume content against a provided Job Description (JD), 
    providing overall fit and keyword density scores. (ENHANCED)
    """
    
    # Retrieve the top N chunks most relevant to the JD
    top_chunks = resume_vectorstore.similarity_search(jd_text, k=20) 
    resume_context = "\n\n".join([doc.page_content for doc in top_chunks])
    
    # Craft the detailed prompt for the LLM to perform the matching analysis
    match_prompt = f"""
    You are an expert HR Analyst. Your goal is to evaluate the candidate's resume (Source: {resume_name}) against the specific Job Description (provided in 'Job Description').

    Perform the analysis based ONLY on the provided resume context and the JD.

    Your output must be structured, clear, and highly focused:

    1.  **Match Score:** Give a percentage score (X/100) indicating the overall fit.
    2.  **Matching Skills:** List the top 5-7 skills or experiences from the RESUME that directly address requirements in the JD.
    3.  **Gaps & Recommendations:** List 3 specific skills or keywords from the JD that are missing or weak in the RESUME, and briefly suggest what could be added to improve the score.
    4.  **Keyword Density Score:** Estimate a second score (Y/100) based purely on how many key JD keywords appear in the resume context.

    ---
    Job Description (JD):
    {jd_text}
    ---
    
    Resume Context (Relevant Snippets from {resume_name}):
    {resume_context}
    ---

    **Format your final response using clear Markdown headings and bullet points:**
    ## 🎯 JD Match Analysis
    
    **Resume Source:** {resume_name}
    
    **Overall Match Score:** X/100
    **Keyword Density Score:** Y/100
    
    ### ✅ Key Matching Skills & Experience
    * [Skill 1 from Resume]
    * [Skill 2 from Resume]
    * ...

    ### ❌ Gaps & Improvement Suggestions
    * [JD Requirement 1]: [Brief suggestion]
    * [JD Requirement 2]: [Brief suggestion]
    * ...
    """
    
    return ask_perplexity(match_prompt)

def generate_optimized_resume(jd_text, resume_vectorstore, resume_name):
    """
    Generates a resume tailored to the JD. (ENHANCED: Focus on result-driven, quantifiable, and JD-aligned bullet points.)
    """
    
    # Retrieve the top N chunks most relevant to the JD
    top_chunks = resume_vectorstore.similarity_search(jd_text, k=30) 
    resume_context = "\n\n".join([doc.page_content for doc in top_chunks])
    
    # Craft the detailed prompt for the LLM to perform the resume generation
    gen_prompt = f"""
    You are a professional Resume Writer specializing in optimizing resumes for Applicant Tracking Systems (ATS) and human recruiters.
    Your task is to generate a new, optimized resume draft for the candidate based on their existing experience (provided in 'Resume Context') and the target Job Description (JD).

    Follow these rules strictly:
    1.  **Tailoring:** Maximize the use of keywords, concepts, and required skills found in the Job Description.
    2.  **Source Material:** ONLY use facts, experience, and accomplishments present in the 'Resume Context'. Do not invent new jobs, dates, or projects.
    3.  **Formatting & Style:** Generate a clean, professional, and easy-to-copy Markdown output. Avoid excessive white space and maintain a high-density, professional resume structure.

    ---
    Job Description (JD):
    {jd_text}
    ---
    
    Resume Context (Relevant Snippets from Source Resume: {resume_name}):
    {resume_context}
    ---

    **Generate the Optimized Resume Draft Now:**
    
    ## [Candidate Name - Inferred from Context]
    * [Inferred Contact Info, e.g., Email | LinkedIn]
    
    ### 🎯 Professional Summary (4-5 strong, tailored sentences)
    [Write a summary that directly integrates 3-4 key JD requirements using the candidate's background and focuses on the most relevant career highlights.]
    
    ### ⚙️ Technical Skills (Prioritize JD Keywords)
    * [Skill Category 1]: [JD Keyword 1], [JD Keyword 2], [Related Skill 3], ...
    * [Skill Category 2]: [JD Keyword 4], [JD Keyword 5], [Related Skill 6], ...
    
    ### 🏢 Professional Experience
    * **[Most Relevant Role]** | [Company] | [Dates]
        * **IMPACT & RESULTS:** Rewrite each bullet point to start with a strong action verb and focus on **quantifiable achievements**. Use numbers, percentages, currency, or scale to demonstrate the business impact of your work, specifically targeting the key responsibilities in the JD.
        * [Tailored, result-driven bullet point 1, e.g.: "Led the migration of 5 high-traffic services to AWS Lambda, resulting in a 30% reduction in operating costs."]
        * [Tailored, result-driven bullet point 2: Highly relevant to JD, focused on measurable success.]
        * [Tailored, result-driven bullet point 3: Highly relevant to JD, focused on measurable success.]
    
    * **[Second Relevant Role]** | [Company] | [Dates]
        * [Result-driven bullet point 1]
        * [Result-driven bullet point 2]
    
    ... [Continue with other relevant experience sections, focusing on JD alignment]
    """
    
    return ask_perplexity(gen_prompt)

def generate_interview_prep(jd_text, resume_vectorstore, resume_name):
    """
    Generates tailored STAR method responses based on the JD requirements and
    the content of the source resume. (ENHANCED)
    """
    
    behavioral_keywords = "team leadership, conflict resolution, problem-solving under pressure, managing deadlines, cross-functional collaboration, overcoming technical challenges"
    
    # Retrieve chunks relevant to both the JD and the behavioral skills
    top_chunks = resume_vectorstore.similarity_search(jd_text + " " + behavioral_keywords, k=25) 
    resume_context = "\n\n".join([doc.page_content for doc in top_chunks])
    
    prep_prompt = f"""
    You are an AI Interview Coach. Your task is to generate three high-quality behavioral interview answers tailored for the role described in the Job Description (JD), using only the facts and experiences provided in the 'Resume Context'.

    **Focus on three common behavioral categories relevant to the JD:**
    1.  A challenge or conflict.
    2.  A project demonstrating leadership/initiative.
    3.  A time when you applied a key technical skill (e.g., Python, Cloud, ML) to solve a critical business problem.

    **Structure each answer STRICTLY using the STAR method (Situation, Task, Action, Result) in a paragraph format.** Do not use bullet points within the STAR paragraphs.

    ---
    Job Description (JD):
    {jd_text}
    ---
    
    Resume Context (Relevant Snippets from Source Resume: {resume_name}):
    {resume_context}
    ---

    **Generate the three tailored STAR method answers now:**
    
    ## 🎙️ Behavioral Interview Prep (STAR Method)
    
    ### 1. Challenge/Conflict Example (Tailored to JD)
    **Situation/Task:** [Describe a situation from the resume where you faced a significant challenge or conflict.]
    **Action:** [Detail the specific steps you took to address the challenge, emphasizing JD-related skills.]
    **Result:** [Quantify the positive outcome or what you learned.]

    ### 2. Leadership/Initiative Example (Tailored to JD)
    **Situation/Task:** [Describe a project or role from the resume where you demonstrated leadership or initiative.]
    **Action:** [Detail the specific actions you took to lead the effort, manage the team, or start the initiative.]
    **Result:** [Quantify the impact on the project or organization.]
    
    ### 3. Key Technical Skill Application Example
    **Situation/Task:** [Describe a time you used a key technical skill (inferred from context) to solve a problem.]
    **Action:** [Detail the implementation, emphasizing technical rigor and best practices.]
    **Result:** [Quantify the resulting efficiency gain, cost saving, or performance improvement.]
    """
    
    return ask_perplexity(prep_prompt)


# ==========================
# CORE CHAT LOGIC FUNCTION 
# ==========================

def handle_chat_logic(vectorstore_key, messages_key):
    """
    Handles the display and processing logic for a single chat interface.
    """
    
    vectorstore = st.session_state.get(vectorstore_key)
    messages = st.session_state.get(messages_key, [])

    # 1. Display Past Messages
    chat_placeholder = st.empty()

    # Chat container height is set to 400px to keep the input bar close
    with chat_placeholder.container(height=400, border=True): 
        for message in messages:
            role = message["role"]
            avatar = "👤" if role == "user" else "🤖"
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(message["content"], unsafe_allow_html=True)
                
    # 2. Check Prerequisites 
    if vectorstore is None:
        if messages_key == DHANESH_MESSAGES_KEY:
             st.info(f"⬆️ **Waiting for file.** The file '{DHANESH_FILE_PATH}' is not yet loaded.")
        else:
             st.info("⬆️ **Waiting for a document.** Please upload a file to start RAG analysis.")
        return

    # 3. Handle Chat Input
    prompt_key = f"prompt_{messages_key}"
    
    if prompt := st.chat_input("Ask a question about the document...", key=prompt_key):
        
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        st.session_state[PROCESSING_STATE_KEY] = messages_key
        st.rerun()

    # 4. Generate Assistant Response 
    if st.session_state.get(PROCESSING_STATE_KEY) == messages_key:
        
        del st.session_state[PROCESSING_STATE_KEY]
        user_prompt = st.session_state[messages_key][-1]["content"]
        
        with st.chat_message("assistant", avatar="🤖"):
            spinner_text = "Searching and synthesizing document content..."
            with st.spinner(spinner_text):
                
                # --- RAG and LLM Logic ---
                rag_start_time = time.time()
                search_progress = st.progress(0, text="1/3: Retrieving document chunks...")
                docs = vectorstore.similarity_search(user_prompt, k=TOP_K_DOCUMENTS)
                context = "\n\n".join([doc.page_content for doc in docs])
                rag_time = time.time() - rag_start_time
                search_progress.progress(33, text="2/3: Formatting history and context...")

                history_for_llm = messages[-MAX_HISTORY_MESSAGES:-1] 
                history = "\n".join([f'{m["role"].capitalize()}: {m["content"]}' for m in history_for_llm])
                
                full_prompt = f"""
                You are a professional, helpful assistant. Your task is to use the provided Context and the recent Conversation History to answer the User's question.
                Only use the given context. Do not invent information (hallucinate). If the context does not contain the answer, state that you cannot find the information in the document.
                
                ---
                Document Context (Retrieved Relevant Snippets):
                {context}
                ---
                Conversation History (for context - most recent messages only):
                {history}
                ---
                User's Current Question:
                {user_prompt}
                """
                
                search_progress.progress(66, text="3/3: Generating and refining response...")
                llm_start_time = time.time()
                answer = ask_perplexity(full_prompt)
                llm_time = time.time() - llm_start_time

                search_progress.empty()
                
                st.markdown(answer)
                
                with st.expander("🔬 RAG Analysis & Source Context"):
                    st.caption(f"**⏱️ Metrics:** Retrieval: **{rag_time:.2f}s** | LLM Call: **{llm_time:.2f}s** | Chunks Retrieved: **{len(docs)}**")
                    st.markdown("---")
                    st.markdown("**Source Chunks:**")
                    for i, doc in enumerate(docs):
                        page_num = doc.metadata.get('page', 'N/A')
                        if page_num != 'N/A': page_num += 1
                        st.markdown(f"**Chunk {i+1}** (Page **{page_num}**):")
                        st.code(doc.page_content[:400] + "...", language='text')

            st.session_state[messages_key].append({"role": "assistant", "content": answer})
            st.rerun()


# ==========================
# MAIN STREAMLIT APP
# ==========================

st.set_page_config(
    page_title="AI Document RAG Assistant", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown(f"""
    <style>
    /* 1. GLOBAL BACKGROUND AND TEXT COLORS */
    .stApp {{
        background-color: {BACKGROUND_DARK}; 
        color: {TEXT_DARK}; 
    }}
    :root {{
        color-scheme: dark; 
    }}
    .st-emotion-cache-18ni4x7 {{ 
        background-color: {BACKGROUND_DARK}; 
        color: {TEXT_DARK}; 
    }}

    /* 2. SIDEBAR (LIGHT BACKGROUND) */
    .st-emotion-cache-1txytx0, [data-testid="stSidebar"] {{ 
        background-color: {BACKGROUND_MEDIUM} !important; 
        color: {TEXT_LIGHT} !important; 
    }}
    .st-emotion-cache-1txytx0 *, [data-testid="stSidebar"] * {{ 
        color: {TEXT_LIGHT} !important; 
    }}
    .st-emotion-cache-1hwfwsd {{ 
        color: {TEXT_LIGHT} !important; 
    }}
    
    /* 3. General Containers/Blocks in main area */
    .st-emotion-cache-1r4y2ny {{ 
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        background-color: {BACKGROUND_DARK}; 
        border: 1px solid #333333; 
    }}

    /* 4. Main Content Headings */
    h1, h2, h3, h4, h5, h6, p, li, div {{
        color: {TEXT_DARK} !important; 
    }}

    /* 5. Chat Messages */
    .st-emotion-cache-4oyq3x.e1g8pov64 {{ /* User Chat */
        background-color: #2c2c2c; 
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: {TEXT_DARK};
    }}
    .st-emotion-cache-1c7yb1q.e1g8pov64 {{ /* Assistant Chat */
        background-color: #111111; 
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #333333;
        color: {TEXT_DARK};
    }}
    
    /* 6. BUTTONS (Accent) */
    .stButton > button {{
        background-color: {PRIMARY_ACCENT};
        border: 1px solid {PRIMARY_ACCENT};
        font-weight: bold;
    }}
    .stButton > button[kind="primary"] > div > span {{
        color: {BACKGROUND_DARK} !important; 
    }}
    .stButton > button:hover {{
        background-color: #0077e6; 
        border-color: #0077e6;
    }}

    /* 7. Expander/Code Blocks (Main area) */
    code {{
        background-color: #333333;
        color: #ff9999; 
    }}
    .stCodeBlock {{
        background-color: #333333;
        border: 1px solid #555555;
    }}
    .stExpander {{
        background-color: #2c2c2c;
        border: 1px solid #444444;
    }}
    .stExpander * {{
        color: {TEXT_DARK} !important;
    }}
    
    /* 8. Sidebar Expander Text Fix */
    [data-testid="stSidebar"] .stExpander * {{
        color: {TEXT_LIGHT} !important; 
    }}

    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 🌟 INITIALIZATION
# =========================================================================

st.title("📚 Intelligent Document Analysis Assistant")
st.markdown(
    f"""
    <p style='color: #cccccc;'>
    **Welcome!** This application offers three powerful ways to analyze PDF documents using <strong>Retrieval-Augmented Generation (RAG)</strong>.
    Choose a mode below to start analyzing.
    </p>
    """, unsafe_allow_html=True
)
st.markdown("---") 

# 2. Initialization Logic (Ensuring all keys are present)
if DHANESH_MESSAGES_KEY not in st.session_state:
    st.session_state[DHANESH_MESSAGES_KEY] = [{"role": "assistant", "content": f"Hello! Welcome to my **Personalized Assistant**. I can answer any questions you have about my background, skills, and experience detailed in **`{DHANESH_FILE_PATH}`**."}]
if GENERAL_MESSAGES_KEY not in st.session_state:
    st.session_state[GENERAL_MESSAGES_KEY] = []
if GENERAL_VECTORSTORE_KEY not in st.session_state:
    st.session_state[GENERAL_VECTORSTORE_KEY] = None
if GENERAL_FILE_HASH_KEY not in st.session_state:
    st.session_state[GENERAL_FILE_HASH_KEY] = None
if DHANESH_VECTORSTORE_KEY not in st.session_state:
    st.session_state[DHANESH_VECTORSTORE_KEY] = None 
if PROCESSING_STATE_KEY not in st.session_state:
    st.session_state[PROCESSING_STATE_KEY] = None 
if GENERAL_SUMMARY_GENERATED not in st.session_state:
    st.session_state[GENERAL_SUMMARY_GENERATED] = False
if JD_TEXT_KEY not in st.session_state:
    st.session_state[JD_TEXT_KEY] = ""
if JD_MATCH_RESULT_KEY not in st.session_state:
    st.session_state[JD_MATCH_RESULT_KEY] = None
if JD_UPLOADED_VECTORSTORE_KEY not in st.session_state:
    st.session_state[JD_UPLOADED_VECTORSTORE_KEY] = None
if JD_UPLOADED_FILE_NAME not in st.session_state:
    st.session_state[JD_UPLOADED_FILE_NAME] = None
if OPTIMIZED_RESUME_RESULT_KEY not in st.session_state: 
    st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = None
if INTERVIEW_PREP_RESULT_KEY not in st.session_state: 
    st.session_state[INTERVIEW_PREP_RESULT_KEY] = None


# --- Load Dhanesh Vector Store on Startup ---
if st.session_state[DHANESH_VECTORSTORE_KEY] is None:
    with st.spinner(f"**Indexing** the local document: **`{DHANESH_FILE_PATH}`**..."):
        st.session_state[DHANESH_VECTORSTORE_KEY] = get_dhanesh_vectorstore()
        
        if st.session_state[DHANESH_VECTORSTORE_KEY]:
             st.session_state[DHANESH_MESSAGES_KEY] = [{"role": "assistant", "content": f"**👋 Ready to Chat!** I'm loaded with all my professional details. Ask me about my projects, experience, or technical skills!"}] 
        
    if not st.session_state.get(PROCESSING_STATE_KEY) and st.session_state[DHANESH_VECTORSTORE_KEY] is not None:
        st.rerun()


# --- Custom Sidebar/Config ---
with st.sidebar:
    st.header("⚙️ App Configuration")
    
    with st.expander("🛠️ RAG & LLM Parameters", expanded=False):
        st.caption("Settings for the Retrieval-Augmented Generation system.")
        st.markdown(f"* **LLM:** `{MODEL}`")
        st.markdown(f"* **RAG K (Chunks):** `{TOP_K_DOCUMENTS}`")
        st.markdown(f"* **Chunk Size:** `{CHUNK_SIZE}`")
        st.markdown(f"* **Chunk Overlap:** `{CHUNK_OVERLAP}`")

    st.markdown("---")
    
    st.markdown("### 🗑️ Clear Data")
    
    if st.button("Clear Dhanesh Chat", use_container_width=True, help="Clear the conversation in the Dhanesh tab."):
        st.session_state[DHANESH_MESSAGES_KEY] = [{"role": "assistant", "content": f"Chat cleared. Ready to analyze `{DHANESH_FILE_PATH}`."}]
        st.session_state[PROCESSING_STATE_KEY] = None
        st.rerun()
        
    if st.button("Clear General RAG", use_container_width=True, help="Clear the conversation and document in the General RAG tab."):
        st.session_state[GENERAL_MESSAGES_KEY] = []
        st.session_state[GENERAL_SUMMARY_GENERATED] = False 
        st.session_state[GENERAL_VECTORSTORE_KEY] = None
        st.session_state[GENERAL_FILE_HASH_KEY] = None
        st.session_state[PROCESSING_STATE_KEY] = None
        st.rerun()
        
    if st.button("Clear JD Tools", use_container_width=True, help="Clear the Job Description text, analysis, and generated resume/prep."):
        st.session_state[JD_TEXT_KEY] = ""
        st.session_state[JD_MATCH_RESULT_KEY] = None
        st.session_state[JD_UPLOADED_VECTORSTORE_KEY] = None 
        st.session_state[JD_UPLOADED_FILE_NAME] = None       
        st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = None 
        st.session_state[INTERVIEW_PREP_RESULT_KEY] = None 
        st.session_state[PROCESSING_STATE_KEY] = None
        st.rerun()

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["🤵 Personalized Document: Meet Dhanesh", "☁️ General Document Uploader", "🎯 JD Match Analyzer / Generator"])

# ==========================
# TAB 1: Dhanesh Resume Analysis 
# ==========================

with tab1:
    st.header("Want to know more about Dhanesh's background?")
    st.caption(f"This assistant is powered by my resume.") 
    handle_chat_logic(DHANESH_VECTORSTORE_KEY, DHANESH_MESSAGES_KEY)


# ==========================
# TAB 2: General RAG Analysis (Upload)
# ==========================

with tab2:
    st.header("Upload and Analyze a New Document")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF document here:", 
        type=["pdf"], 
        key="general_uploader", 
        help="The file will be chunked, embedded, and indexed for chat analysis."
    )

    if uploaded_file:
        if st.session_state[GENERAL_FILE_HASH_KEY] != uploaded_file.file_id or st.session_state[GENERAL_VECTORSTORE_KEY] is None:
            
            st.session_state[GENERAL_MESSAGES_KEY] = []
            st.session_state[GENERAL_FILE_HASH_KEY] = uploaded_file.file_id
            st.session_state[GENERAL_SUMMARY_GENERATED] = False 
            
            file_content = uploaded_file.read()
            
            with st.spinner(f"**Indexing** document: '{uploaded_file.name}'..."):
                st.session_state[GENERAL_VECTORSTORE_KEY] = get_vectorstore(BytesIO(file_content))
            
            if st.session_state[GENERAL_VECTORSTORE_KEY]:
                st.session_state[GENERAL_MESSAGES_KEY].append({
                    "role": "assistant",
                    "content": f"**🎉 Success!** Document **`{uploaded_file.name}`** is loaded. Now analyzing the content..."
                })
            else:
                st.error("❌ Failed to process the document. Please try a different PDF.")
            
            st.rerun()

    if st.session_state[GENERAL_VECTORSTORE_KEY] is not None and not st.session_state[GENERAL_SUMMARY_GENERATED]:
        
        with st.chat_message("assistant", avatar="🤖"):
            summary_placeholder = st.empty()
            summary_placeholder.markdown("*(Running initial document analysis...)*")
            
            llm_start_time = time.time()
            summary_result = generate_document_summary(st.session_state[GENERAL_VECTORSTORE_KEY], uploaded_file.name if uploaded_file else "Uploaded Document")
            llm_time = time.time() - llm_start_time
            
            final_message = f"""
            **📄 Document Analysis Complete!** (Time: {llm_time:.2f}s)
            
            {summary_result}
            
            ---
            **You can now ask questions about the document content!**
            """
            
            st.session_state[GENERAL_MESSAGES_KEY].append({
                "role": "assistant",
                "content": final_message
            })
            
            st.session_state[GENERAL_SUMMARY_GENERATED] = True
            st.rerun()
            
    handle_chat_logic(GENERAL_VECTORSTORE_KEY, GENERAL_MESSAGES_KEY)


# ==========================
# TAB 3: JD Match Analyzer / Generator
# ==========================

with tab3:
    st.header("Resume-to-JD Matching and Optimization Tool")
    st.caption("Analyze any resume against a JD and generate an optimized resume draft.")
    
    # --- Resume Selector and Uploader ---
    st.subheader("1. Select Your Resume Source")
    
    uploaded_jd_file = st.file_uploader(
        "Upload a Resume PDF to analyze (Optional):", 
        type=["pdf"], 
        key="jd_resume_uploader",
        help="Upload a new resume PDF to check against the JD. If left blank, the default resume is used."
    )

    # Process uploaded file if a new one is provided for JD match
    if uploaded_jd_file:
        if st.session_state.get(JD_UPLOADED_FILE_NAME) != uploaded_jd_file.name or st.session_state.get(JD_UPLOADED_VECTORSTORE_KEY) is None:
            
            st.session_state[JD_UPLOADED_FILE_NAME] = uploaded_jd_file.name
            
            file_content = uploaded_jd_file.read()
            
            with st.spinner(f"**Indexing** uploaded resume: '{uploaded_jd_file.name}'..."):
                st.session_state[JD_UPLOADED_VECTORSTORE_KEY] = get_vectorstore(BytesIO(file_content))
            
            if st.session_state[JD_UPLOADED_VECTORSTORE_KEY]:
                st.success(f"✅ Successfully loaded **`{uploaded_jd_file.name}`** for analysis.")
                st.session_state[JD_MATCH_RESULT_KEY] = None 
                st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = None 
                st.session_state[INTERVIEW_PREP_RESULT_KEY] = None 
            else:
                st.error("❌ Failed to process the uploaded resume.")
            st.rerun()
            
    # Determine which resume to use
    default_vs = st.session_state.get(DHANESH_VECTORSTORE_KEY)
    uploaded_vs = st.session_state.get(JD_UPLOADED_VECTORSTORE_KEY)
    
    if uploaded_vs is not None:
        source_vs = uploaded_vs
        source_name = st.session_state[JD_UPLOADED_FILE_NAME]
        st.info(f"➡️ Currently analyzing **Uploaded Resume:** `{source_name}`")
    elif default_vs is not None:
        source_vs = default_vs
        source_name = DHANESH_FILE_PATH
        st.info(f"➡️ Currently analyzing **Default Resume:** `{source_name}`")
    else:
        source_vs = None
        source_name = None
        st.warning("⚠️ No resume is currently loaded. Please ensure the default file exists or upload a resume.")
        
    st.markdown("---")
    
    # --- JD Input and Analysis ---
    st.subheader("2. Paste Job Description")

    jd_input = st.text_area(
        "Paste the Full Job Description (JD) here:",
        key=JD_TEXT_KEY,
        height=300,
        placeholder="E.g., Senior Software Engineer required with 5+ years of Python, AWS, and Machine Learning experience...",
        value=st.session_state[JD_TEXT_KEY]
    )
    
    can_analyze = source_vs is not None and jd_input
    
    # BUTTONS - 3 columns for all tools
    col_analyze, col_generate, col_interview = st.columns(3)
    
    with col_analyze:
        # Analyze Button
        if st.button(f"🚀 Analyze JD Match", use_container_width=True, disabled=(not can_analyze)):
            st.session_state[JD_MATCH_RESULT_KEY] = None 
            st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = None 
            st.session_state[INTERVIEW_PREP_RESULT_KEY] = None 
            
            with st.spinner(f"Processing Job Description and analyzing resume: {source_name}..."):
                analysis_start_time = time.time()
                result = analyze_jd_match(jd_input, source_vs, source_name)
                analysis_time = time.time() - analysis_start_time
                
                # Store the result
                st.session_state[JD_MATCH_RESULT_KEY] = {
                    "output": result,
                    "time": analysis_time
                }
            st.rerun()
            
    with col_generate:
        # Generate Resume Button
        if st.button(f"📝 Generate Optimized Resume", use_container_width=True, disabled=(not can_analyze)):
            st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = None 
            
            with st.spinner(f"Generating optimized resume draft based on JD and {source_name}..."):
                gen_start_time = time.time()
                result = generate_optimized_resume(jd_input, source_vs, source_name)
                gen_time = time.time() - gen_start_time
                
                # Store the result
                st.session_state[OPTIMIZED_RESUME_RESULT_KEY] = {
                    "output": result,
                    "time": gen_time
                }
            st.rerun()

    with col_interview: 
        # Generate Interview Prep Button
        if st.button(f"🗣️ Generate Interview Prep", use_container_width=True, disabled=(not can_analyze)):
            st.session_state[INTERVIEW_PREP_RESULT_KEY] = None 
            
            with st.spinner(f"Generating tailored interview answers based on JD and {source_name}..."):
                prep_start_time = time.time()
                result = generate_interview_prep(jd_input, source_vs, source_name)
                prep_time = time.time() - prep_start_time
                
                # Store the result
                st.session_state[INTERVIEW_PREP_RESULT_KEY] = {
                    "output": result,
                    "time": prep_time
                }
            st.rerun()
        
    st.markdown("---")

    # Display Results
    st.subheader("3. Results")
    
    # Display JD Match Analysis Report
    if st.session_state.get(JD_MATCH_RESULT_KEY):
        result_data = st.session_state[JD_MATCH_RESULT_KEY]
        
        st.markdown("### 📊 JD Match Analysis Report")
        
        # Display score and analysis (Attempt to extract both scores)
        try:
            match_overall = re.search(r'\*\*Overall Match Score:\*\* (\d+)/100', result_data["output"])
            score_overall = int(match_overall.group(1)) if match_overall else None
            
            match_keyword = re.search(r'\*\*Keyword Density Score:\*\* (\d+)/100', result_data["output"])
            score_keyword = int(match_keyword.group(1)) if match_keyword else None
            
            if score_overall is not None:
                st.metric(label="Overall Match Score", value=f"{score_overall}%", delta="Analysis Complete")
                st.progress(score_overall, text=f"Overall Confidence: {score_overall}%")
                
            if score_keyword is not None:
                st.caption(f"Keyword Density Score: **{score_keyword}%**")
                
        except Exception:
             pass 

        st.markdown(result_data["output"])
        st.caption(f"Analysis completed in **{result_data['time']:.2f}s**.")
        st.markdown("***") 
        
    # Display Optimized Resume Draft
    if st.session_state.get(OPTIMIZED_RESUME_RESULT_KEY):
        gen_data = st.session_state[OPTIMIZED_RESUME_RESULT_KEY]
        
        st.markdown("### 📝 Generated Optimized Resume Draft")
        st.markdown(
            f"""
            <p style='color: #ffcc00; font-size: 14px;'>
            💡 **ACTION REQUIRED:** This is a draft. Please review, verify personal details, and refine formatting before use.
            </p>
            """, unsafe_allow_html=True
        )

        st.text_area(
            "Copy Optimized Resume Draft (Markdown Format)", 
            value=gen_data["output"], 
            height=600,
            key="optimized_resume_output"
        )
        st.caption(f"Generation completed in **{gen_data['time']:.2f}s**.")
        st.markdown("***")

    # Display Interview Prep Results
    if st.session_state.get(INTERVIEW_PREP_RESULT_KEY):
        prep_data = st.session_state[INTERVIEW_PREP_RESULT_KEY]
        
        st.markdown("### 🗣️ Behavioral Interview Prep (STAR Method)")
        st.markdown(
            f"""
            <p style='color: #ffcc00; font-size: 14px;'>
            💡 **REMINDER:** These answers are based only on the provided resume context. You should rehearse them naturally.
            </p>
            """, unsafe_allow_html=True
        )

        st.markdown(prep_data["output"])
        st.caption(f"Preparation generated in **{prep_data['time']:.2f}s**.")
        st.markdown("***")