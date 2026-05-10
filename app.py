import streamlit as st
import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag import load_cv, create_vector_store, analyze_job_description
from agent import analyze_match, generate_cover_letter

st.set_page_config(
    page_title = "Job Application Agent", 
    page_icon = "🤖", 
    layout = "wide"
)

st.title("🤖 Job application Intellegence Agent")
st.markdown("Upload your CV and paste a job description...")

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
        with open("data/temp_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing your CV against the job description"):
            chunks = load_cv("data/temp_cv.pdf")
            vectorstore = create_vector_store(chunks)
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