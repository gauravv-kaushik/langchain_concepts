from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4', temperature=0.9, max_completion_tokens=1000)
# temperature is the creativity of the response, higher value means more creative response and every time you run the code you will get different response, if you set temperature to 0 then you will get same response every time you run the code. min value of temperature is 0 and max value is 1. default value of temperature is 0.7

# max_completion_tokens is the maximum number of tokens that the model can generate in response to a prompt.


res = model.invoke("What is the capital of India?")
print(res.content)