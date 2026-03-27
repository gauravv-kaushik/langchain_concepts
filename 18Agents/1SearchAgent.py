from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent, AgentState


search_tool = DuckDuckGoSearchRun()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1)

agent = create_agent(model=model, tools=[search_tool])

res = agent.invoke({
    "messages": [
        {"role": "user", "content": "give me three ways to go to goa from delhi"}
    ]
})

print(res["messages"][-1].content[0]['text'])
