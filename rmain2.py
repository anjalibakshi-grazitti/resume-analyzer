import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import json
from typing import Dict, List, Optional, Any
import pandas as pd
import re
import time
from datetime import datetime
from pdfminer.high_level import extract_text

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .score-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .feedback-section {
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .collapsible {
        background-color: #f1f1f1;
        color: #444;
        cursor: pointer;
        padding: 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        border-radius: 5px;
        margin: 5px 0;
    }
    .collapsible:hover {
        background-color: #ddd;
    }
    .content {
        padding: 0 18px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PdfMinerLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        text = extract_text(self.file_path)
        doc = type('Document', (), {'page_content': text, 'metadata': {'source': self.file_path}})()
        return [doc]

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

def generate_feedback(resume_vectorstore, job_description_text, resume_name):
    retriever = resume_vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4 mini", temperature=0.1)

    prompt = f"""
You are an expert HR professional and career advisor with 15+ years of experience in technical recruitment. 
Your task is to evaluate the following resume against the provided job description.

Job Description:
\"\"\"
{job_description_text}
\"\"\"

Resume Name: {resume_name}

Please provide a detailed analysis following this EXACT format:

SCORE: [Provide an ATS score from 1-100, where 100 is perfect match. Consider:
- Keyword matching (30 points)
- Skills alignment (25 points)
- Experience relevance (25 points)
- Education match (10 points)
- Overall presentation (10 points)]

Technical Skills Match:
1. [First skill match]
2. [Second skill match]
3. [Third skill match]
List all technical skills found in the resume that match the job requirements.

STRENGTHS:
1. [First strength]
2. [Second strength]
3. [Third strength]
Provide the strengths in a bullet point format.

WEAKNESSES:
1. [First weakness]
2. [Second weakness]
3. [Third weakness]
Provide the weaknesses in a bullet point format.

RECOMMENDATIONS:
1. [First recommendation]
2. [Second recommendation]
3. [Third recommendation]
Provide the recommendations in a bullet point format.

FINAL VERDICT:
[Write a clear recommendation on whether to proceed with this candidate (YES/NO) and a brief justification]

Note: Be specific and provide concrete examples from both the resume and job description in your analysis.
Focus on technical skills, experience alignment, and potential for success in the role.
"""

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    result = chain({"query": prompt})
    return result["result"]

def extract_score(feedback_text):
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", feedback_text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return min(max(score, 1), 100)  # ATS score is out of 100
    return -1

def parse_resume_to_json(resume_text: str) -> Dict[str, Any]:
    """Parse resume text into structured JSON format using GPT-4"""
    prompt = f"""
    Parse the following resume into a structured JSON format. Extract the following information:
    1. Personal Information:
       - Full Name
       - Email
       - Phone
       - Location (if available)
       - LinkedIn (if available)
       - GitHub/Portfolio (if available)
    
    2. Education:
       - List of education entries with:
         * Degree
         * Institution
         * Year
         * GPA (if available)
         * Relevant coursework (if available)
    
    3. Experience:
       - List of work experiences with:
         * Company
         * Position
         * Duration
         * Key responsibilities and achievements
         * Technologies used
    
    4. Skills:
       - Technical Skills (categorized)
       - Soft Skills
       - Tools and Technologies
       - Certifications
    
    5. Projects (if available):
       - Project name
       - Description
       - Technologies used
       - Duration
       - Key achievements
    
    Resume Text:
    \"\"\"
    {resume_text}
    \"\"\"
    
    Return the information in a valid JSON format with the following structure:
    {{
        "personal_info": {{
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "linkedin": "",
            "github": ""
        }},
        "education": [
            {{
                "degree": "",
                "institution": "",
                "year": "",
                "gpa": "",
                "coursework": []
            }}
        ],
        "experience": [
            {{
                "company": "",
                "position": "",
                "duration": "",
                "responsibilities": [],
                "technologies": []
            }}
        ],
        "skills": {{
            "technical": [],
            "soft": [],
            "tools": [],
            "certifications": []
        }},
        "projects": [
            {{
                "name": "",
                "description": "",
                "technologies": [],
                "duration": "",
                "achievements": []
            }}
        ]
    }}
    
    Note: If any field is not found in the resume, leave it as an empty string or empty array.
    Ensure the output is valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a resume parser that extracts structured information and returns it in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Clean the response to ensure it's valid JSON
        # Remove any markdown code block indicators
        result = re.sub(r'```json\n?|\n?```', '', result)
        
        # Parse the JSON
        parsed_data = json.loads(result)
        return parsed_data
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return None

def evaluate_resume(resume_path, jd_text, resume_name):
    resume_docs = load_document(resume_path)
    if not resume_docs:
        return None, -1, None
        
    # Extract structured data from resume
    resume_text = "\n".join([doc.page_content for doc in resume_docs])
    structured_data = parse_resume_to_json(resume_text)
    
    resume_vectorstore = create_retriever(resume_docs)
    feedback = generate_feedback(resume_vectorstore, jd_text, resume_name)
    score = extract_score(feedback)
    
    return feedback, score, structured_data

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
                    score_color = "green"
                    score_label = "Excellent Match"
                elif score >= 60:
                    score_color = "orange"
                    score_label = "Good Match"
                else:
                    score_color = "red"
                    score_label = "Needs Improvement"
                
                st.markdown(f"""
                    <div class="score-box">
                        <h4>ATS Score: <span style="color: {score_color}">{score}/100</span></h4>
                        <p style="color: {score_color}">{score_label}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with st.expander("View Detailed Feedback", expanded=False):
                st.markdown("""
                    <div class="feedback-section">
                        <h4>Detailed Analysis:</h4>
                """, unsafe_allow_html=True)
                
                st.markdown(res["feedback"])
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")

def save_to_csv(results: List[Dict], output_path: str = "resume_eval.csv") -> str:
    """
    Save resume evaluation results to CSV file with specified columns.
    If file exists, append new results while avoiding duplicates.
    """
    try:
        # Prepare data for CSV
        csv_data = []
        for res in results:
            structured_data = res.get('structured_data', {})
            personal_info = structured_data.get('personal_info', {})
            
            # Format contact information
            contact_info = []
            if personal_info.get('phone'):
                contact_info.append(f"Phone: {personal_info['phone']}")
            if personal_info.get('location'):
                contact_info.append(f"Location: {personal_info['location']}")
            if personal_info.get('linkedin'):
                contact_info.append(f"LinkedIn: {personal_info['linkedin']}")
            if personal_info.get('github'):
                contact_info.append(f"GitHub: {personal_info['github']}")
            
            contact_str = " | ".join(contact_info) if contact_info else "N/A"
            
            # Create row data
            row = {                
                'name': personal_info.get('name', 'N/A'),
                'contact': contact_str,
                'email': personal_info.get('email', 'N/A'),
                'feedback': res.get('feedback', 'N/A'),
                'ats_score': res.get('score', 'N/A'),
                'resume_file': res.get('name', 'N/A')
            }
            csv_data.append(row)
            
        # Create DataFrame
        df_new = pd.DataFrame(csv_data)
            
        # Check if file exists and append if it does
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            # Combine old and new data, dropping duplicates based on resume_file 
            df_combined = pd.concat([df_existing, df_new]).drop_duplicates(
                subset=['resume_file'],
                keep='last'
            )
            df_combined.to_csv(output_path, index=False)
        else:
            df_new.to_csv(output_path, index=False)
            
        return output_path

    except Exception as e:
        st.error(f"Error saving to CSV: {str(e)}")
        return None

def main():
    st.title("📄 Resume Evaluator")
    
    st.markdown("""
    ### AI-Powered Resume Analysis
    Upload multiple resumes and a job description to get detailed analysis and scoring.
    """)

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
        st.markdown("- PDF files")
        st.markdown("- Word documents (DOCX)")
        st.markdown("- Text files (TXT)")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Files")
        resume_files = st.file_uploader(
            "Upload Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload one or more resumes to evaluate"
        )
        
    with col2:
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            help="Upload the job description to compare against"
        )

    if resume_files and jd_file:
        top_n = st.slider(
            "Select number of top candidates to display",
            min_value=1,
            max_value=len(resume_files),
            value=min(3, len(resume_files))
        )

        if st.button("🔍 Start Evaluation", type="primary"):
            with st.spinner("Analyzing resumes..."):
                progress_bar = st.progress(0)
                
                jd_path = f"temp_jd.{jd_file.name.split('.')[-1]}"
                with open(jd_path, "wb") as f:
                    f.write(jd_file.read())

                jd_docs = load_document(jd_path)
                if not jd_docs:
                    st.error("Error processing job description. Please try again.")
                    return
                    
                jd_text = "\n".join([doc.page_content for doc in jd_docs])
                results = []

                for i, resume_file in enumerate(resume_files):
                    progress_bar.progress((i + 1) / len(resume_files))
                    
                    resume_path = f"temp_resume_{resume_file.name}"
                    with open(resume_path, "wb") as f:
                        f.write(resume_file.read())

                    feedback, score, structured_data = evaluate_resume(resume_path, jd_text, resume_file.name)
                    
                    if feedback and score != -1:
                        results.append({
                            "name": resume_file.name,
                            "score": score,
                            "feedback": feedback,
                            "structured_data": structured_data
                        })

                    os.remove(resume_path)

                os.remove(jd_path)
                progress_bar.empty()

                if results:
                    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
                    display_results(sorted_results, top_n)
                    
                    # Export to CSV
                    csv_path = save_to_csv(sorted_results)
                    if csv_path:
                        st.success(f"Data exported to CSV: {csv_path}")
                        
                        # Display preview of the CSV data
                        df = pd.read_csv(csv_path)
                        st.markdown("### 📊 CSV Data Preview")
                        st.dataframe(df)
                    
                    # Also export to JSON for detailed data
                    export_data = {
                        
                        "job_description": jd_text,
                        "candidates": [
                            {
                                "resume_name": res["name"],
                                "score": res["score"],
                                "structured_data": res["structured_data"]
                            }
                            for res in sorted_results
                        ]
                    }
                    
                    json_path = "resume_evaluations.json"
                    with open(json_path, "w") as f:
                        json.dump(export_data, f, indent=2)
                    
                    st.success(f"Detailed data exported to JSON: {json_path}")
                else:
                    st.error("No valid results were generated. Please try again.")
    else:
        st.info("👆 Please upload at least one resume and a job description to begin evaluation.")

if __name__ == "__main__":
    main()


 
