from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model='claude-3-5-sonnet-20240909', temperature=0.9, max_completion_tokens=1000)

res = model.invoke("What is the capital of India?")
print(res.content)