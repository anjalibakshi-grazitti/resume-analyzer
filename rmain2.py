# ===============================================================
# Resume Evaluator - Refined & Fully Compatible With requirements.txt
# ===============================================================

# Standard Libraries
import os
import json
import re
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import streamlit as st
from dotenv import load_dotenv

# OpenAI (new official client)
from openai import OpenAI

# PDF Extractor
from pdfminer.high_level import extract_text

# LangChain Imports (Latest 0.2+ Split Packages)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ===============================================================
# Setup & Config
# ===============================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# PDF Loader (Custom Fallback)
# ===============================================================

class PdfMinerLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        text = extract_text(self.file_path)
        return [Document(page_content=text, metadata={"source": self.file_path})]

# ===============================================================
# Document Loader
# ===============================================================

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

# ===============================================================
# Vector Store (FAISS)
# ===============================================================

def create_retriever(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# ===============================================================
# Feedback Generator Using Retrieval + GPT
# ===============================================================

def generate_feedback(resume_vectorstore, job_description, resume_name):
    retriever = resume_vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

    prompt = f"""
You are an expert HR professional with 15+ years of experience.

Evaluate the resume named "{resume_name}" against the job description below.

Job Description:
\"\"\"{job_description}\"\"\"

Follow this format STRICTLY:

SCORE: <1-100 ATS score>

Technical Skills Match:
1.
2.
3.

STRENGTHS:
1.
2.
3.

WEAKNESSES:
1.
2.
3.

RECOMMENDATIONS:
1.
2.
3.

FINAL VERDICT:
(YES/NO + 2 lines justification)
"""

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    result = chain({"query": prompt})
    return result["result"]

def extract_score(text):
    match = re.search(r"SCORE:\s*(\d+)", text)
    if match:
        score = float(match.group(1))
        return min(max(score, 1), 100)
    return -1

# ===============================================================
# Resume → JSON Parser Using GPT
# ===============================================================

def parse_resume_to_json(text: str) -> Dict[str, Any]:
    prompt = f"""
Extract structured JSON from this resume text:

\"\"\"{text}\"\"\"

Return valid JSON only with fields:
personal_info, education, experience, skills, projects.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        result = response.choices[0].message.content
        result = re.sub(r'```json|```', '', result).strip()

        return json.loads(result)

    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return {}

# ===============================================================
# Evaluate Resume
# ===============================================================

def evaluate_resume(resume_path, jd_text, filename):
    docs = load_document(resume_path)
    if not docs:
        return None, -1, {}

    full_text = "\n".join([d.page_content for d in docs])
    parsed_json = parse_resume_to_json(full_text)

    vectorstore = create_retriever(docs)
    feedback = generate_feedback(vectorstore, jd_text, filename)
    score = extract_score(feedback)

    return feedback, score, parsed_json

# ===============================================================
# Display Results
# ===============================================================

def display_results(sorted_results, top_n):
    st.markdown("## 📊 Evaluation Results")

    for idx, res in enumerate(sorted_results[:top_n], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### 📄 {idx}. {res['name']}")

                with st.expander("View Structured Data", False):
                    st.json(res["structured_data"])

            with col2:
                score = res["score"]

                color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                label = "Excellent Match" if score >= 80 else "Good Match" if score >= 60 else "Needs Improvement"

                st.markdown(f"""
                    <div class="score-box">
                        <h4>ATS Score: <span style="color:{color}">{score}/100</span></h4>
                        <p style="color:{color}">{label}</p>
                    </div>
                """, unsafe_allow_html=True)

            with st.expander("Detailed Feedback"):
                st.write(res["feedback"])

            st.markdown("---")

# ===============================================================
# Save CSV
# ===============================================================

def save_to_csv(results, filename="resume_eval.csv"):
    try:
        rows = []
        for res in results:
            pi = res["structured_data"].get("personal_info", {})

            rows.append({
                "name": pi.get("name", ""),
                "email": pi.get("email", ""),
                "score": res["score"],
                "resume_file": res["name"],
                "feedback": res["feedback"]
            })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return filename

    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")
        return None

# ===============================================================
# Streamlit UI
# ===============================================================

def main():
    st.title("📄 AI-Powered Resume Evaluator")

    col1, col2 = st.columns(2)
    with col1:
        resumes = st.file_uploader("Upload Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
    with col2:
        jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])

    if resumes and jd_file:
        if st.button("Start Evaluation"):
            # Save JD
            jd_path = f"temp_jd.{jd_file.name.split('.')[-1]}"
            with open(jd_path, "wb") as f:
                f.write(jd_file.read())

            jd_docs = load_document(jd_path)
            jd_text = "\n".join([d.page_content for d in jd_docs])

            results = []
            for r in resumes:
                path = f"temp_{r.name}"
                with open(path, "wb") as f:
                    f.write(r.read())

                feedback, score, parsed = evaluate_resume(path, jd_text, r.name)

                if feedback:
                    results.append({
                        "name": r.name,
                        "feedback": feedback,
                        "score": score,
                        "structured_data": parsed
                    })

                os.remove(path)

            os.remove(jd_path)

            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            display_results(sorted_results, top_n=min(3, len(sorted_results)))

            csv_path = save_to_csv(sorted_results)
            if csv_path:
                st.success(f"CSV saved: {csv_path}")
                st.dataframe(pd.read_csv(csv_path))

# ===============================================================
if __name__ == "__main__":
    main()
