from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

class Contact(TypedDict):
    full_name:Annotated[str, "The full name of the contact"]
    email:Annotated[Optional[str], "The email address of the contact"]
    phone:Annotated[Optional[str], "The phone number of the contact"]
    dob:Annotated[Optional[str], "The date of birth of the contact in YYYY-MM-DD format if year is not known then use 2000 as year"]
    anniversary:Annotated[Optional[str], "The anniversary date of the contact in YYYY-MM-DD format if year is not known then use 2000 as year"]
    children:Annotated[Optional[list[object]], "The list of children of the contact {name:str, birthday:str in YYYY-MM-DD format if year is not known then use 2000 as year}"]
    employment:Annotated[Optional[list[object]], "The list of employment details of the contact {employer name, details}"]
    interests:Annotated[Optional[list[str]], "The list of interests of the contact"]


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

structured_model = model.with_structured_output(Contact)

res = structured_model.invoke("create a contact name gaurav kaushik his email is gaurav@gmail.com and his birthday comes on 12 nov and anniversary on 10 dec he has two children rohan born on 2012-05-10 and rhea born on 2015-08-20 he has worked at google as software engineer and at amazon as senior software engineer his interests are coding, music and traveling")

print(res)