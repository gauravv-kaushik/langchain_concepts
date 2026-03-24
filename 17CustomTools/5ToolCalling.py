from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()


@tool
def multiply(a:int, b:int)->int:
    """To multiply two given numbers a and b"""
    return a*b

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.4)

llm_with_tools = model.bind_tools([multiply])

res = llm_with_tools.invoke("multiply 6 with 4") #LLM can't call tools itself, it just suggests which tool to call

if res.tool_calls:
    tool_call = res.tool_calls[0]
    result = multiply.invoke(tool_call)
    print(result)