from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

schema = [
    ResponseSchema(name="fact1", description="Fact1 about the topic"),
    ResponseSchema(name="fact2", description="Fact2 about the topic"),
    ResponseSchema(name="fact3", description="Fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"topic": "black hole"})

model_response = model.invoke(prompt)

parsed_response = parser.parse(model_response.content)

print(parsed_response)

# using chain
chain = template | model | parser
final_res = chain.invoke({"topic": "black hole"})
print(final_res)