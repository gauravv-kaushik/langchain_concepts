from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

detail_prompt = PromptTemplate(
    template="Generate a deatiled report on {topic}",
    input_variables=["topic"]
    )

summary_prompt = PromptTemplate(
    template="Summarize the 5 pointer summary on following report: \n{report}",
    input_variables=["report"]
    )

parser = StrOutputParser()

chain = detail_prompt | model | parser | summary_prompt | model | parser

res = chain.invoke({"topic": "AI negative impact on society"})
print(res)

chain.get_graph().print_ascii()