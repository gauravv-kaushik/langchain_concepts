from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

prompt1 = PromptTemplate(
    template="Generate a tweet on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post on {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)

parallel_chain = RunnableParallel({
    "tweet":chain1,
    "linkedin":chain2
})

res = parallel_chain.invoke({"topic":"AI impact on Developers"})

print(res)