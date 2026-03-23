from langchain.tools import tool
# from langchain_core.tools import tool


@tool
def multiply(a:int, b:int)-> int:
    """Multiply two numbers"""
    return a*b

res = multiply.invoke({"a":3, "b":5})
print(res)
print(multiply.name)
print(multiply.description)
print(multiply.args)