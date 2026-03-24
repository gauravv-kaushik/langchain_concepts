from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import requests
import json
load_dotenv()


#https://app.exchangerate-api.com/   -> Get API KEY from this website

exchange_rate_api_key = "dcd845a76d5c5e880f1f5bb2"

from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def get_conversion_factor(base_currency: str, target_currency: str)-> float:
    """This function fetches the currency conversion factor between a given base currency and a target currency"""

    url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()

@tool
def convert_currency(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg])-> float:
    """This function calculates the target currency value for a given currency value by multiplying with conversion rate"""
    return base_currency_value * conversion_rate



llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.4)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])

query = HumanMessage("What is conversion rate factor between USD to INR and based on that rate can you convert 10 dollar to INR")
messages = [query]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

conversion_rate = None

for tool_call in ai_message.tool_calls:
    if tool_call['name'] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call['args'])
        data = tool_message1
        conversion_rate = data['conversion_rate']
        messages.append(tool_message1)

# 👉 Now FORCE second tool call yourself
if conversion_rate:
    result = convert_currency.invoke({
        "base_currency_value": 10,
        "conversion_rate": conversion_rate
    })
    print(result)



