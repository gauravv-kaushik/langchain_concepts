from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

prompt1 = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke :\n{joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

joke_generation = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "words":RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_generation,parallel_chain)
res = final_chain.invoke({"topic":"HR"})

print(res)