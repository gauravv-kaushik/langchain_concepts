import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# video_id = "Gfr50f6ZBvo"
video_id = input("Enter video ID (not complete URL i.e Gfr50f6ZBvo) : ") # Gfr50f6ZBvo
api = YouTubeTranscriptApi()

try:
    transcript_list = api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No caption available in this video")

splitter  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.2)

prompt = PromptTemplate(
    template="""
            You're a helpful Assistant.
            Answer ONLY from the provided transcripted context.
            If the context is insufficient , Just say you don't know in a polite way.
            {context}\n
            Question: {question}
        """,
    input_variables=["context", "question"]
    )


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

while(True):
    query = input("\nQuestion : ")
    res = main_chain.invoke(query)
    print("Answer : ",res,"\n")
