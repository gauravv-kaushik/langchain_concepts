from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Langchain helps developers build LLM application easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models"),
]

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

query = "What is chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"content:\n{doc.page_content}")