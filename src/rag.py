from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

def load_cv(pdf_path : str):
    print(f"Load PDF: {pdf_path}")

    loader = PDFPlumberLoader(pdf_path)
    pages = loader.load()

    print(f"Pages loaded: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(pages)

    print(f"Chunks created: {len(chunks)}")

    return chunks

def create_vector_store(chunks):

    load_dotenv()
    print("Creating vector database from data")


    #use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/gemini-embedding-001"
    )


    #create ChromaDB
    vectorstore = Chroma.from_documents(
        documents = chunks, 
        embedding = embeddings, 
        persist_directory ="data/chroma_db"
    )

    print("Database was created")

    return vectorstore

def search_cv(vectorstore, query:str, k: int = 3):
    results = vectorstore.similarity_search(query, k = k)

    return results

def analyze_job_description(job_text:str, vectorstore) -> dict:
    cv_results = search_cv(vectorstore, job_text, k=5)

    cv_context = "\n\n".join([doc.page_content for doc in cv_results])

    return {
        "job_text" : job_text, 
        "cv_context": cv_context
    }


