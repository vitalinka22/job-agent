from fastapi import FastAPI, File, UploadFile, Form 
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.dirname(__file__))

from rag import load_cv, create_vector_store, analyze_job_description
from agent import analyze_match, generate_cover_letter

app =FastAPI(
    title = "Job Application Agent API",
    description = "API for analyzing job descriptions and generating cover letters",
    version = "1.0"
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"], 
    allow_methods = ["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Job Application Agent API is running!"}

@app.post("/analyze")
async def analyze(
    cv_file: UploadFile = File(...), 
    job_description : str = Form(...)
):
    cv_path =f"data/temp_{cv_file.filename}"
    with open(cv_path, "wb") as f:
        f.write(await cv_file.read())

    chunks = load_cv(cv_path)
    vectorstore = create_vector_store(chunks)
    result = analyze_job_description(job_description, vectorstore)
    analysis = analyze_match(result["job_text"], result["cv_context"])
    cover_letter = generate_cover_letter(result["job_text"], result["cv_context"])

    return {
        "match_analysis": analysis["analysis"], 
        "cover_letter": cover_letter
    }

