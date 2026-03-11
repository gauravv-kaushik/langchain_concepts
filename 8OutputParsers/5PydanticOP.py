from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)


class Person(BaseModel):
    name: Annotated[str, Field(...,description="Name of the person")]
    age: int = Field(...,gt=18, description="Age of the person")
    city: Optional[str] = Field(None, description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="give me the name, age and city of a fictional {country} person \n {format_instruction}",
    input_variables=["country"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.invoke({"country": "Indian"})
model_response = model.invoke(prompt)
final_res = parser.parse(model_response.content)
print(final_res)

# using chain
chain = template | model | parser
final_res = chain.invoke({"country": "Britisher"})
print(final_res)