from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Annotated, Optional, List, Dict

load_dotenv()

class Contact(BaseModel):
    name: Annotated[str, Field(..., min_length=2,description="The full name of the contact", examples=["Gaurav Kaushik"])]

    email: Annotated[Optional[EmailStr], Field(None, description="The email address of the contact")]

    phone: Annotated[Optional[str], Field(None, description="The phone number of the contact")]

    dob: Annotated[Optional[str], Field(None, description="The date of birth of the contact in YYYY-MM-DD format if year is not known then use 2000 as year", examples=["1990-05-10", "2000-05-10"])]

    anniversary: Annotated[Optional[str], Field(None, description="The anniversary date of the contact in YYYY-MM-DD format if year is not known then use 2000 as year", examples=["1990-05-10", "2000-05-10"])]

    children: Annotated[Optional[List[Dict[str, str]]], Field(None, description="The list of children of the contact ", examples=[{"name": "Rohan", "birthday": "2012-05-10"}, {"name": "Rhea", "birthday": "2015-08-20"}])]

    employment: Annotated[Optional[List[Dict[str, str]]], Field(None, description="The list of employment details of the contact", examples=[{"employer": "Google", "details": "Software Engineer"}, {"employer": "Amazon", "details": "Senior Software Engineer"}])]

    interests: Annotated[Optional[List[str]], Field(None, description="The list of interests of the contact")]

    @field_validator("dob", "anniversary", mode="before")
    def validate_date(cls, value):
        if value is None:
            return value
        if value.count("-") == 1:
            return "2000-" + value
        return value
    
    @field_validator("phone", mode="before")
    def validate_phone(cls, value):
        if value is None:
            return value
        cleaned = ''.join(c for c in value if c.isdigit() or c == '+' or c == ' ')
        if len(cleaned) != 10:
            raise ValueError("Phone number must be 10 digits long")
        return cleaned
    

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

structured_model = model.with_structured_output(Contact)

res = structured_model.invoke("create a contact name gaurav kaushik his email is gaurav@gmail.com and mobie no. is 9876543210 and his birthday comes on 12 nov and anniversary on 10 dec he has two children rohan born on 2012-05-10 and rhea born on 2015-08-20 he has worked at google as software engineer and at amazon as senior software engineer his interests are coding, music and traveling")

print(type(res.model_dump()))
print(type(res.model_dump_json()))
print(res)