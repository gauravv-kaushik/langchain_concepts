import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Langchain helps developers build LLM application easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models"),
]


embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding
)

retriever = vectorstore.as_retriever(
    search_type='mmr',    # MMR -> Maximum Marginal Relevance 
    search_kwargs={"k":3, "lambda_mult":0.5} # k = Top Results, lambda_mult = relevance-diversity balance (0 to 5)
)

query = "What is Langchain ?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"content:\n{doc.page_content}")
