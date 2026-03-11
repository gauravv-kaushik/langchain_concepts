from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


#chat Template
chat_template  = ChatPromptTemplate([
    ('system', "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{query}"),
])

#load chat history
chat_history = []
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())


prompt = chat_template.invoke({'query': 'Where is my refund?', 'chat_history': chat_history})


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

res = model.invoke(prompt)
print(res.content)