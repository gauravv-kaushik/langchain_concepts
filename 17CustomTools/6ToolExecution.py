from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()


@tool
def multiply(a:int, b:int)->int:
    """To multiply two given numbers a and b"""
    return a*b

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.4)

llm_with_tools = model.bind_tools([multiply])

query = HumanMessage("What is product of 3 and 4")

messages = [query]

res = llm_with_tools.invoke(messages)

messages.append(res)

tool_result = multiply.invoke(res.tool_calls[0])

messages.append(tool_result)

print(messages)

final_res = llm_with_tools.invoke(messages)

print(final_res.content)
