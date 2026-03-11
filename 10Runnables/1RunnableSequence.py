from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

prompt = PromptTemplate(
    template="Write a short joke on {topic}",
    input_variables=['topic']
)
parser = StrOutputParser()

explain_prompt = PromptTemplate(
    template="write the joke and explaination below for the following joke \n {joke}",
    input_variables=['joke']
)

chain = RunnableSequence(prompt, model, parser, explain_prompt, model, parser)

res = chain.invoke({"topic":"Developers"})
print(res)