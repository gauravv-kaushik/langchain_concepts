from langchain_community.document_loaders import TextLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

prompt = PromptTemplate(
    template="summarize the following text:\n{text}",
    input_variables=['text']
)

parser = StrOutputParser()


chain = prompt | model | parser

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

res = chain.invoke({'text':docs[0].page_content})

print(res)