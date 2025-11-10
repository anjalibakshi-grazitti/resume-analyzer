import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import json
from typing import Dict, List, Any
import pandas as pd
import re
from datetime import datetime
from pdfminer.high_level import extract_text

load_dotenv()

# --- Streamlit UI setup ---
st.set_page_config(
    page_title="Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%; background-color: #4CAF50; color: white;
        padding: 0.5rem 1rem; border-radius: 5px; border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    .score-box {
        padding: 1rem; border-radius: 5px; background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .feedback-section {
        padding: 1rem; border-left: 4px solid #4CAF50; margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Custom PDF loader using pdfminer ---
class PdfMinerLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        text = extract_text(self.file_path)
        doc = type('Document', (), {'page_content': text, 'metadata': {'source': self.file_path}})()
        return [doc]


# --- Document loader wrapper ---
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PdfMinerLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type!")
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None


# --- Split, embed, and create retriever ---
def create_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store


# --- Generate AI feedback for a resume ---
def generate_feedback(resume_vectorstore, job_description_text, resume_name):
    retriever = resume_vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    prompt_template = ChatPromptTemplate.from_template("""
You are an expert HR professional and career advisor with 15+ years of experience in technical recruitment.
Your task is to evaluate the following resume against the provided job description.

Job Description:
\"\"\"{job_description}\"\"\"

Resume Name: {resume_name}

Please provide a detailed analysis following this EXACT format:

SCORE: [Provide an ATS score from 1-100, where 100 is perfect match. Consider:
- Keyword matching (30 points)
- Skills alignment (25 points)
- Experience relevance (25 points)
- Education match (10 points)
- Overall presentation (10 points)]

Technical Skills Match:
1. [Skill 1]
2. [Skill 2]
3. [Skill 3]

STRENGTHS:
1. [Strength 1]
2. [Strength 2]
3. [Strength 3]

WEAKNESSES:
1. [Weakness 1]
2. [Weakness 2]
3. [Weakness 3]

RECOMMENDATIONS:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

FINAL VERDICT:
[YES/NO with justification]
""")

    combine_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, combine_chain)

    result = retrieval_chain.invoke({
        "input": f"Evaluate resume {resume_name} based on the job description.",
        "job_description": job_description_text,
        "resume_name": resume_name
    })

    return result.get("answer", "No feedback generated.")


# --- Extract ATS score from feedback ---
def extract_score(feedback_text):
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", feedback_text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return min(max(score, 1), 100)
    return -1


# --- Parse resume to JSON ---
def parse_resume_to_json(resume_text: str) -> Dict[str, Any]:
    prompt = f"""
    Parse the following resume into a structured JSON format with fields for
    personal_info, education, experience, skills, and projects.
    Resume:
    \"\"\"{resume_text}\"\"\"
    Return valid JSON.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract structured resume information in JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        result = re.sub(r'```json\n?|\n?```', '', response.choices[0].message.content)
        return json.loads(result)
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return None


# --- Evaluate resume end-to-end ---
def evaluate_resume(resume_path, jd_text, resume_name):
    resume_docs = load_document(resume_path)
    if not resume_docs:
        return None, -1, None

    resume_text = "\n".join([doc.page_content for doc in resume_docs])
    structured_data = parse_resume_to_json(resume_text)
    resume_vectorstore = create_retriever(resume_docs)
    feedback = generate_feedback(resume_vectorstore, jd_text, resume_name)
    score = extract_score(feedback)
    return feedback, score, structured_data


# --- Display results ---
def display_results(sorted_results, top_n):
    st.markdown("## 📊 Evaluation Results")
    for idx, res in enumerate(sorted_results[:top_n], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 📄 {idx}. {res['name']}")
                if res.get('structured_data'):
                    with st.expander("View Structured Resume Data", expanded=False):
                        st.json(res['structured_data'])
            with col2:
                score = res['score']
                if score >= 80:
                    color, label = "green", "Excellent Match"
                elif score >= 60:
                    color, label = "orange", "Good Match"
                else:
                    color, label = "red", "Needs Improvement"
                st.markdown(f"""
                    <div class="score-box">
                        <h4>ATS Score: <span style="color:{color}">{score}/100</span></h4>
                        <p style="color:{color}">{label}</p>
                    </div>
                """, unsafe_allow_html=True)
            with st.expander("View Detailed Feedback", expanded=False):
                st.markdown(f"""
                    <div class="feedback-section">
                        <h4>Detailed Analysis:</h4>
                        {res['feedback']}
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---")


# --- Save to CSV ---
def save_to_csv(results: List[Dict], output_path: str = "resume_eval.csv") -> str:
    try:
        csv_data = []
        for res in results:
            sd = res.get('structured_data', {})
            pi = sd.get('personal_info', {})
            contact_info = " | ".join(filter(None, [
                f"Phone: {pi.get('phone', '')}" if pi.get('phone') else "",
                f"Location: {pi.get('location', '')}" if pi.get('location') else "",
                f"LinkedIn: {pi.get('linkedin', '')}" if pi.get('linkedin') else "",
                f"GitHub: {pi.get('github', '')}" if pi.get('github') else ""
            ]))
            csv_data.append({
                "name": pi.get('name', 'N/A'),
                "email": pi.get('email', 'N/A'),
                "contact": contact_info or "N/A",
                "feedback": res.get('feedback', 'N/A'),
                "ats_score": res.get('score', 'N/A'),
                "resume_file": res.get('name', 'N/A')
            })
        df_new = pd.DataFrame(csv_data)
        if os.path.exists(output_path):
            df_old = pd.read_csv(output_path)
            df = pd.concat([df_old, df_new]).drop_duplicates(subset=['resume_file'], keep='last')
        else:
            df = df_new
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        st.error(f"Error saving to CSV: {str(e)}")
        return None


# --- Main Streamlit app ---
def main():
    st.title("📄 Resume Evaluator")
    st.markdown("### AI-Powered Resume Analysis")

    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool uses advanced AI to:
        - Score resumes against job descriptions  
        - Provide detailed feedback  
        - Extract structured resume data  
        - Identify strengths and weaknesses  
        - Give actionable recommendations
        """)
        st.markdown("### 📋 Supported Formats")
        st.markdown("- PDF files\n- Word documents (DOCX)\n- Text files (TXT)")

    col1, col2 = st.columns(2)
    with col1:
        resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    with col2:
        jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])

    if resume_files and jd_file:
        top_n = st.slider("Select number of top candidates", 1, len(resume_files), min(3, len(resume_files)))
        if st.button("🔍 Start Evaluation", type="primary"):
            with st.spinner("Analyzing resumes..."):
                progress = st.progress(0)
                jd_path = f"temp_jd.{jd_file.name.split('.')[-1]}"
                with open(jd_path, "wb") as f:
                    f.write(jd_file.read())
                jd_docs = load_document(jd_path)
                if not jd_docs:
                    st.error("Error reading job description.")
                    return
                jd_text = "\n".join([doc.page_content for doc in jd_docs])
                results = []
                for i, resume in enumerate(resume_files):
                    progress.progress((i + 1) / len(resume_files))
                    path = f"temp_resume_{resume.name}"
                    with open(path, "wb") as f:
                        f.write(resume.read())
                    feedback, score, data = evaluate_resume(path, jd_text, resume.name)
                    if feedback and score != -1:
                        results.append({"name": resume.name, "score": score, "feedback": feedback, "structured_data": data})
                    os.remove(path)
                os.remove(jd_path)
                progress.empty()
                if results:
                    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
                    display_results(sorted_results, top_n)
                    csv_path = save_to_csv(sorted_results)
                    if csv_path:
                        st.success(f"Results exported to: {csv_path}")
                        st.dataframe(pd.read_csv(csv_path))
                else:
                    st.error("No valid results generated.")
    else:
        st.info("👆 Please upload resumes and a job description to start.")


if __name__ == "__main__":
    main()
