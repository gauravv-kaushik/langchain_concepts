from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(...,description="whether the sentiment of feedback is positive or negative")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

feedback_prompt = PromptTemplate(
    template="Classify the sentiment of following feedback whether positive or negative: \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":pydantic_parser.get_format_instructions()}
)

positive_prompt = PromptTemplate(
    template="write appropriate response to this positive feedback : \n{feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="write appropriate response to this negative feedback : \n{feedback}",
    input_variables=["feedback"]
)


sentiment_chain = feedback_prompt | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", positive_prompt | model | parser),
    (lambda x: x.sentiment == "Negative", negative_prompt | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = sentiment_chain | branch_chain

res = chain.invoke({"feedback":"My name is Gaurav"})
print(res)


