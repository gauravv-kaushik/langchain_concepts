from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.4
)


notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text: \n {text}",
    input_variables=["text"]
)

quiz_prompt = PromptTemplate(
    template="Generate 5 short question answers from the following text with number format: \n {text}",
    input_variables=["text"]
)

merge_prompt = PromptTemplate(
    template="""
        Combine the following notes and quiz.

        IMPORTANT:
        - Keep all 5 question-answers exactly as they are.
        - Do NOT rewrite or summarize the quiz.
        - Maintain number format.

        Notes:
        {notes}

        Quiz:
        {quiz}
    """,
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
   {
    "notes": notes_prompt | model1 | parser,
    "quiz": quiz_prompt | model2 | parser
   }
)

merged_chain =  merge_prompt | model1 | parser

chain = parallel_chain | merged_chain

text = """
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression that works by finding the optimal hyperplane which separates data points of different classes with the maximum possible margin. The key idea is to choose a decision boundary that not only separates the classes but also maximizes the distance between the closest points (called support vectors) and the hyperplane, making the model more robust and less prone to overfitting. In cases where data is not linearly separable, SVM uses a soft margin approach with slack variables to allow some misclassification, and it can also apply the kernel trick (such as linear, polynomial, or RBF kernels) to transform data into higher-dimensional space where a linear separator becomes possible. Because of its ability to handle high-dimensional data and complex boundaries effectively, SVM is widely used in applications like text classification, image recognition, and bioinformatics.
"""

res = chain.invoke({"text": text})

print(res)

chain.get_graph().print_ascii()