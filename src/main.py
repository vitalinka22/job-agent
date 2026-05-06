from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os 
import sys

sys.path.append(os.path.dirname(__file__))
from rag import load_cv, create_vector_store, analyze_job_description

load_dotenv()

JOB_TEXT = """
We are looking for a Working Student AI Engineer.
Requirements: Python, LangChain, RAG systems, Azure,
LLM experience, FastAPI, Docker, Git.
"""

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError ("GOOGLE_API_KEY was not found")

llm = ChatGoogleGenerativeAI(model ="gemini-2.5-flash")

def main():
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, "data", "cv.pdf")
    chunks = load_cv(pdf_path)
    vectorstore = create_vector_store(chunks)

    result = analyze_job_description(JOB_TEXT, vectorstore)
    print(result["cv_context"])

if __name__ == "__main__":
    main()