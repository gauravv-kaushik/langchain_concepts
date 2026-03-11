from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic} and response must be in numbered format. without **",
    input_variables=["topic"]
    )

parser = StrOutputParser()

chain = prompt | model | parser # Langchain expression Language (LCEL) -> Pipeline

chain.get_graph().print_ascii() # -> to visualize the chain

res = chain.invoke({"topic": "space"})
print(res)