from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

load_dotenv()

template = PromptTemplate(
    template="What year did {company} was founded and who is the CEO of {company}?",
    input_variables=["company"],
    validate_template=True
    )

# prompt = template.invoke({'company': 'Amazon'})

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)


chain = template | model #chain the prompt template and the model together

res = chain.invoke({'company': 'Amazon'})

print(res.content)
