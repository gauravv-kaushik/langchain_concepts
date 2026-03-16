from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')


text_splitter = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

text = """
Artificial Intelligence is transforming modern technology. Machine learning models can analyze large datasets and make predictions with high accuracy. Deep learning, which is a subset of machine learning, uses neural networks with multiple layers to learn complex patterns in data. Many companies use AI for recommendation systems, fraud detection, and natural language processing.

Python is one of the most popular programming languages for artificial intelligence and data science. Libraries such as NumPy, pandas, TensorFlow, and PyTorch make it easier to build machine learning models. Developers also use frameworks like Django and FastAPI to build backend APIs that serve machine learning models to applications.

Cloud computing has also become an important part of modern software development. Platforms like AWS, Google Cloud, and Microsoft Azure allow developers to deploy applications without managing physical servers. Services such as EC2, S3, and Lambda enable scalable and cost-efficient infrastructure for startups and enterprises.

Mobile app development is growing rapidly as well. Frameworks like React Native allow developers to build cross-platform applications using JavaScript and React. Many modern apps integrate APIs, authentication systems, push notifications, and real-time chat features.

Cybersecurity is another critical area in technology. Organizations must protect their systems from threats such as phishing attacks, malware, and data breaches. Implementing strong authentication, encryption, and secure coding practices helps reduce security risks in software systems.
"""

docs = text_splitter.create_documents([text])
print(docs)
print(len(docs))