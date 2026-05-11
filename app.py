import streamlit as st
import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag import load_cv, create_vector_store, analyze_job_description
from agent import analyze_match, generate_cover_letter

@st.cache_resource
def get_vectorstore(file_path: str):
    chunks = load_cv(file_path)
    vectorstore = create_vector_store(chunks)
    return vectorstore

st.set_page_config(
    page_title = "Job Application Agent", 
    page_icon = "🤖", 
    layout = "wide"
)

st.title("🤖 Job application Intellegence Agent")
st.markdown("Upload your CV and paste a job description...")

with st.sidebar:
    st.header("📖 How to use")
    st.markdown("""
    1. Upload your CV as PDF
    2. Paste the job description
    3. Click **Analyze Match**
    4. Get your match score and cover letter!
    """)
    st.divider()
    st.markdown("Built with LangChain + Gemini + ChromaDB")


col1, col2 = st.columns(2)

with col1 :
    st.header("📄 Your CV")
    uploaded_file = st.file_uploader("Upload CV (PDF)", type = "pdf")

with col2:
    st.header("💼 Job Description") 
    job_text = st.text_area(
        "Paste job ddescription here", 
        height = 300, 
        placeholder = "Paste the job description here..."
    ) 

if st.button("🔍 Analyze Match", type = "primary"):
    if not uploaded_file:
        st.error("Please upload your CV first!")
    elif not job_text:
        st.error("Please paste a job description")
    else:
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing your CV against the job description"):
            vectorstore =get_vectorstore(file_path)
            result = analyze_job_description(job_text, vectorstore)
            analysis = analyze_match(job_text, result["cv_context"])
            cover_letter = generate_cover_letter(job_text, result["cv_context"])

            st.success("Analysis complete!")

            col3, col4 = st.columns(2)

            with col3:
                st.header("📊 Match Analysis")
                st.markdown(analysis["analysis"])

            with col4:
                st.header("✉️ Cover Letter")
                st.markdown(cover_letter)

                st.download_button(
                    label="📥 Download Cover Letter", 
                    data=cover_letter, 
                    file_name="cover_letter.txt", 
                    mime="text/plain"
                )