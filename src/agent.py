from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

def analyze_match(job_text: str, cv_context: str) -> dict:
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    prompt = f"""
    You are a senior recruiter at a top tech company with 10 years of experience.
    
    Carefully analyze how well this candidate's CV matches the job description.
    Be specific and honest - not overly positive.
    
    JOB DESCRIPTION:
    {job_text}
    
    RELEVANT CV SECTIONS:
    {cv_context}
    
    Provide your analysis in this EXACT format:
    
    MATCH SCORE: X%
    (Give a realistic score. 90%+ only for near-perfect matches)
    
    MATCHING SKILLS:
    - skill: brief explanation why it matches
    
    MISSING SKILLS:
    - skill: why it matters for this role
    
    RECOMMENDATIONS:
    - specific action the candidate can take to improve their application
    
    SUMMARY:
    2-3 honest sentences about the overall fit.
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