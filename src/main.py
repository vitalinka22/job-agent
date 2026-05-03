from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError ("GOOGLE_API_KEY was not found")

llm = ChatGoogleGenerativeAI(model ="gemini-2.5-flash")

response = llm.invoke("Hi! Tell me what you can")

print("Gemini is connected sucessfully!")
print(f"Response: {response.content}")