from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write five line summary of following text :\n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic":"black hole"})


res = model.invoke(prompt1)
print(res.content)

prompt2 = template2.invoke({"text":res.content})

res2 = model.invoke(prompt2)
print(res2)