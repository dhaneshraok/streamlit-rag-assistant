📚 Intelligent Document RAG Assistant
This project is a powerful, multi-purpose Streamlit application designed for intelligent document analysis using Retrieval-Augmented Generation (RAG). It integrates LangChain components (FAISS vector store, HuggingFace embeddings) with the Perplexity LLM API (sonar-pro) to provide context-aware chat, resume-to-JD matching, and optimized resume/interview prep generation.

The application is structured into three main modules:

Personalized Document Chat: Chat directly with the hardcoded primary document (resume.pdf).

General Document Uploader: Upload any PDF for a quick RAG Q&A session.

JD Match Analyzer / Generator: The core tool for career optimization, providing match scores, gaps analysis, and optimized resume drafts based on a Job Description.

✨ Features and Modules
Module	Purpose	Key Functionality
🤵 Personalized Chat	Interactive Q&A about a fixed, hardcoded document (resume.pdf).	handle_chat_logic using DHANESH_VECTORSTORE_KEY.
☁️ General Uploader	On-the-fly RAG analysis for any uploaded PDF document.	Dynamic vector store creation (get_vectorstore) and initial document summarization (generate_document_summary).
🎯 JD Match Analyzer	Resume-to-Job-Description analysis and content generation.	analyze_jd_match (Match Score, Gaps), generate_optimized_resume (ATS-friendly draft), generate_interview_prep (STAR answers).
Core RAG System	The engine powering all chat and analysis functions.	Uses LangChain (FAISS, RecursiveCharacterTextSplitter) and Perplexity AI (sonar-pro) for high-quality, grounded responses.
⚙️ Prerequisites
To run this application, you will need:

Python 3.8+

A Perplexity AI API Key.

API Key Setup

The application uses the Perplexity API for all LLM calls. You must set your API key as an environment variable or modify the source code.

Method 1: Environment Variable (Recommended for security)

Bash
export PERPLEXITY_API_KEY="YOUR_ACTUAL_PERPLEXITY_API_KEY"
Method 2: Directly in app.py (Replace the placeholder line)

Python
# app.py (near the top)
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "YOUR_PERPLEXITY_API_KEY_HERE") 
# CHANGE TO:
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "YOUR_ACTUAL_PERPLEXITY_API_KEY") 
🚀 Installation and Local Setup
Clone the Repository

Bash
git clone [Your Repository URL]
cd [repository-folder-name]
Create a Virtual Environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
Install Dependencies

Since your code uses LangChain Community (PyPDFLoader, FAISS, HuggingFaceEmbeddings) and requests for the Perplexity API, you need to ensure these are installed.

Note: Create a requirements.txt file in your project root with the following content:

streamlit
langchain
langchain-community
pypdf
requests
sentence-transformers
Then, run:

Bash
pip install -r requirements.txt
Add Your Default Document Place your primary document (the one for the Personalized Chat tab) in the root directory and ensure it is named as specified in the code:

resume.pdf
💻 Running the Application
Execute the application using Streamlit:

Bash
streamlit run [your_main_app_file].py
(Based on the code structure, the file is likely named app.py or similar.)

The application will launch in your web browser, typically at http://localhost:8501.

🎯 JD Match Analyzer Usage Flow
The most valuable module is the JD Match Analyzer. Here's a quick guide on how to use it:

Navigate to the 🎯 JD Match Analyzer / Generator tab.

Select Resume Source: By default, it uses resume.pdf. You can upload a different resume PDF using the file uploader in step 1 of the tab.

Paste Job Description: Paste the full text of the Job Description into the large text area in step 2.

Analyze & Generate:

Click "Analyze JD Match Score" to generate the Overall Match Score, Keyword Density Score, and detailed Gaps & Recommendations.

Click "Generate Optimized Resume Draft" to get a new draft of your resume, with bullet points rewritten to be results-focused and keyword-aligned with the JD.

Click "Generate Interview Prep" to receive three tailored STAR method answers based on your experience and the JD requirements.

🛠️ Configuration and RAG Parameters
You can adjust key parameters via the ⚙️ App Configuration sidebar on the left:

Parameter	Default Value	Description
MODEL	"sonar-pro"	The Perplexity LLM model used for generation.
CHUNK_SIZE	500	The size of text segments (chunks) created from the document.
CHUNK_OVERLAP	150	The number of tokens shared between adjacent chunks (improves context).
TOP_K_DOCUMENTS	10	The number of most relevant chunks retrieved from the FAISS vector store to pass to the LLM.
MAX_HISTORY_MESSAGES	6	The maximum number of past chat messages sent to the LLM to maintain conversation context.
🗑️ Clear Data Functions
The sidebar includes dedicated buttons to clear the data for each tab, ensuring clean starts for new analyses:

Clear Dhanesh Chat: Resets the conversation history for the Personalized Chat.

Clear General RAG: Clears the chat, removes the loaded document, and resets the vector store for the Uploader tab.

Clear JD Tools: Clears the JD text, all match analysis results, and the uploaded resume (if applicable) in the JD Match tab.
