from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write five line summary of following text :\n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({"topic":"black hole"})

print(res)