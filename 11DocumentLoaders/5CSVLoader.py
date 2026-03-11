from langchain_community.document_loaders import CSVLoader
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
    template="Give me answer of {question} from the following text\n{text}",
    input_variables=['question','text']
)

parser = StrOutputParser()


loader = CSVLoader(file_path='students.csv')

docs = loader.load()

print(docs[0])

chain = prompt | model | parser

res = chain.invoke({'question':'how much percentage kabir got', 'text':docs})
print(res)