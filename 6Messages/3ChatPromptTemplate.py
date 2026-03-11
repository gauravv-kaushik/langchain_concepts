from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} expert."),
#     HumanMessage(content="Explain in simple term, what is {topic}?"),
# ])


# Above code is incorrect


chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain in simple term, what is {topic}?"),
])

prompt = chat_template.invoke({'domain': 'AI', 'topic': 'machine learning'})

print(prompt)
