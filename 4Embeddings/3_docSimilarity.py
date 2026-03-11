from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=32)


docs = ["The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]
query = "What is the capital of France?"

docs_embedding = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)


cosine_similarity_score = cosine_similarity([query_embedding], docs_embedding)[0]

idx, score = sorted(list(enumerate(cosine_similarity_score)), key=lambda x: x[1], reverse=True)[0]
print(f"Most similar document: {docs[idx]} with cosine similarity score: {score}")