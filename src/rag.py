from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

def load_cv(pdf_path : str):
    print(f"Load PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
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


if __name__ == "__main__":
    chunks = load_cv('data/cv.pdf')
    vectorstore = create_vector_store(chunks)

    print("Test search: 'Python experience'")

    result = search_cv(vectorstore, 'Python experience')

    for i, doc in enumerate(result):
        print(f"\n ---Result {i + 1} ----")
        print(doc.page_content)
