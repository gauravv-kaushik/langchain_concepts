from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Annotated

class MultiplyInput(BaseModel):
    a : Annotated[int, Field(..., description="The first number to multiply")]
    b : Annotated[int, Field(..., description="The Secomd number to multiply")]

def multiply(a:int, b:int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

res = multiply_tool.invoke({"a":5,"b":8})

print(res)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)