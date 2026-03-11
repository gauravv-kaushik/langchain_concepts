from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    res = model.invoke(user_input)
    print("AI: ", res.content)