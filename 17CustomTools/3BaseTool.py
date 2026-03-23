from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Annotated

class MultiplyInput(BaseModel):
    a : Annotated[int, Field(..., description="The first number to multiply")]
    b : Annotated[int, Field(..., description="The Secomd number to multiply")]

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self, a:int, b:int) -> int:
        return a*b
    
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a":3, "b":5})

print(result)