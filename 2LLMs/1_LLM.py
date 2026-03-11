from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')
res = llm.invoke("Who is Prime minister of India")
print(res)

llm2 = init_chat_model(model='gpt-4.1')
res2 = llm2.invoke("What is Capital of India")
print(res2)