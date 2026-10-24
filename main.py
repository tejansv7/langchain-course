import os
from dotenv import load_dotenv
#from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Gemini on Vertex AI 
# llm = ChatVertexAI(
#     model="gemini-2.5-flash",
#     project=os.getenv("GOOGLE_CLOUD_PROJECT"),
#     location="us-central1",
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
)

response = llm.invoke("Hello, how are you?")
print("Response from Gemini on Vertex AI:")
print(response.content)
