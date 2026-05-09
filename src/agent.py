from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

def analyze_match(job_text: str, cv_context: str) -> dict:
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    prompt = f"""
    You are an expert career advisor and recruiter.
    
    Analyze how well this candidate's CV matches the job description.
    
    JOB DESCRIPTION:
    {job_text}
    
    RELEVANT CV SECTIONS:
    {cv_context}
    
    Please provide:
    1. MATCH SCORE: A percentage (0-100%) of how well the CV matches
    2. MATCHING SKILLS: List skills from the CV that match the job requirements
    3. MISSING SKILLS: List important skills from the job that are missing in the CV
    4. SUMMARY: 2-3 sentences explaining the match
    
    Format your response exactly like this:
    MATCH SCORE: X%
    
    MATCHING SKILLS:
    - skill 1
    - skill 2
    
    MISSING SKILLS:
    - skill 1
    - skill 2
    
    SUMMARY:
    Your summary here
    """

    response = llm.invoke(prompt)

    return {
        "analysis" : response.content, 
        "job_text": job_text, 
        "cv_context": cv_context
    }

def generate_cover_letter(job_text: str, cv_context : str) -> str:
    """
    Generate a personalized cover letter based on job description and CV.
    """

    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    prompt = f"""
    You a an expert career coach writing a cover letter.

    Write a shor, personalized cover letter (max 3 paragraphs)
    for this candidate based on their CV and the job description.

    JOB DESCRIPTION:
    {job_text}

    RELEVANT CV SECTIONS
    {cv_context}

    Requirements for the cover letter:
    - Sound natural and human, not like a robot
    - Highlight only skills that are actually in the CV
    - Be specific - mention real projects and experience
    - Keep it under 200 words
    - Start with "Dear Hiring Team,"
    - End with "Best regards, Vitalina Alipova"
    - Do NOT make up any experience that is not in the CV
    """

    response = llm.invoke(prompt)

    return response.content