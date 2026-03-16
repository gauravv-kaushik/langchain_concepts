from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is one of the most consistent batsmen in cricket and a former captain of Royal Challengers Bangalore in the IPL.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is known for his elegant batting and leadership. He captains the Mumbai Indians in the IPL.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni is famous for his calm leadership and finishing ability. He led Chennai Super Kings to multiple IPL titles.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is one of the best fast bowlers in modern cricket and a key player for Mumbai Indians.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a world-class all-rounder known for his sharp fielding and match-winning performances for Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='chroma_db',
    collection_name='sample'
)

vectors = vector_store.add_documents(docs)
print("Vectors :\n",vectors)
print()

print("All Documents :\n",vector_store.get(include=['embeddings', 'documents','metadatas']))
print()



print("who among of these are a bowler?\n",vector_store.similarity_search(
    query="who among of these are a bowler?",
    k=2 # how many top results to show
))
print()


print("who among of these are a bowler? with score",vector_store.similarity_search_with_score(
    query="who among of these are a bowler?",
    k=2
))
print()


print("Players with Team Mumbai Indians :\n",vector_store.similarity_search_with_score(
    query="players",
    filter={"team":"Mumbai Indians"}
))
print()


# UPDATE DOCUMENT

doc_updated = Document(
    page_content="Virat Kohli is one of the greatest modern-day batsmen and a key player for Royal Challengers Bengaluru in the IPL, known for his consistency and aggressive batting.",
    metadata={"team": "Royal Challengers Bengaluru"}
)
vector_store.update_document(document_id=vectors[0],document=doc_updated)
print("All Documents after Updation:\n",vector_store.get(include=['embeddings', 'documents','metadatas']))
print()


vector_store.delete(ids=[vectors[4]])
print("All Documents after Deletion:\n",vector_store.get(include=['embeddings', 'documents','metadatas']))
print()

