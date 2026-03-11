from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=8)

result = embedding.embed_query("What is the capital of France?")

print(str(result))