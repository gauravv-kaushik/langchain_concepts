from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


# Without chain

prompt = template.format()

res = model.invoke(prompt)
final_res = parser.parse(res.content)
print(final_res)

# OR using chain

chain = template | model | parser
final_res = chain.invoke({})
print(final_res)