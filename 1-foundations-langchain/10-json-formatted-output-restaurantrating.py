import argparse
from itertools import product
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import json

load_dotenv()  # take environment variables from .env.

class RestaurantReview(BaseModel):
    name: str = Field(..., description="Name of the Restaurant")
    cuisine: str = Field(..., description="Type of cuisine, e.g., 'Italian', 'Chinese', 'Mexican'")
    city: Optional[str] = Field(None, description="City where the restaurant is located. use null if not mentioned")
    rating: Optional[float] = Field(..., description="Rating from 0.0. to 5.0")
    price_range: Optional[str] = Field(None, description="Price range, e.g., 'low', 'mid', 'high'")
    pros: List[str] = Field(default_factory=list, description="List of pros of the Restaurant as bulleted points")
    cons: List[str] = Field(default_factory=list, description="List of cons of the Restaurant as bulleted points")

args = argparse.ArgumentParser(description="JSON Formatted Output with Pydantic")
args.add_argument("--restaurant_name", type=str, required=True, help="Name of the restaurant")
args.add_argument("--cuisine", type=str, required=True, help="Type of cuisine of the restaurant")
args.add_argument("--city", type=str, required=False, help="City where the restaurant is located")
args.add_argument("--rating", type=float, required=False, help="Rating of the restaurant from 0.0 to 5.0")
args.add_argument("--review", type=str, required=False, help="Description of the restaurant review")
args.add_argument("--price_range", type=str, required=False, help="Price range of the restaurant, e.g., 'low', 'mid', 'high'")
args.add_argument("--max_retries", type=int, default=2, help="Maximum number of retries for validation")
parsed_args = args.parse_args()

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    temperature=0.5,
                                    max_tokens=2048)
parser = PydanticOutputParser(pydantic_object=RestaurantReview)    
format_instructions = parser.get_format_instructions()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are an information extraction assistant. Extract Clean restaurant review information from user input.
         if the field is unknown for the provided text, set it to null. 
         Or an empty list (for collections), and never fabricate the facts.
        """),
        ("user",
         """
         Extract restaurant review information from the text below and format it as per the instructions:
         INPUT_NAME: {name}\n
         INPUT_REVIEW: {review}\n
         INPUT_CUISINE: {cuisine}\n
         INPUT_CITY: {city}\n
         INPUT_RATING: {rating}\n
         INPUT_PRICE_RANGE: {price_range}\n
         Follow the format instructions carefully.\n 
         1. Return ONLY VALID JSON that adhere to the format instructions.
         2.Do not include any commentary or markdown fences
         3. Adhere to JSON schema and field descriptions.
         {format_instructions}    
         """
         )
    ]
)

chain = prompt | llm_gemini

variables = {
    "name": parsed_args.restaurant_name,
    "review": parsed_args.review or "No review provided",
    "cuisine": parsed_args.cuisine,
    "city": parsed_args.city or "Not specified", 
    "rating": str(parsed_args.rating) if parsed_args.rating else "Not specified",
    "price_range": parsed_args.price_range or "Not specified",
    "format_instructions": format_instructions
}

last_error_hint = "" # blank for first try

for attempt in range(parsed_args.max_retries + 1):
    if attempt ==0:
        # first attempt : vanilla invoke
        prompt_prepared =  prompt.format(**variables)
    else:
        #retry attempt : vanilla invoke
        retry_prompt = ChatPromptTemplate.from_messages(
            [ *prompt.messages,
             ("user",
              f"""The previous output was invalid JSON.
              Here is the error encountered: {last_error_hint}\n\n
              Please correct the output to adhere to the format instructions.
              Please return ONLY the corrected JSON that follows the exact schema.
              Do not include any extra text"""
              )])       
        prompt_prepared =  retry_prompt.format(**variables)
    
    # invoke the model
    response = llm_gemini.invoke(prompt_prepared)
    print(f"\nResponse from Gemini via LCEL (Attempt {attempt + 1}):")
    print(response.content)
    content = response.content.strip()

    # Remove markdown fences if any
    if content.startswith("```") and content.endswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        content = "\n".join(lines[1:-1]).strip()
    
    try:
        # Quick check for valid JSON
        _ = json.loads(content)
        
        # Parse with Pydantic
        restaurant_review: RestaurantReview = parser.parse(content)
        
        print("\nParsed Restaurant Review Information:")
        print(restaurant_review.model_dump_json(indent=2))
        print("It was successful!")
        break  # exit the retry loop on success
        
    except json.JSONDecodeError as e_json:
        last_error_hint = f"JSON Parsing Error: {str(e_json)}"
        print(f"Validation failed: {last_error_hint}")
        if attempt == parsed_args.max_retries:
            print("Max retries reached. Exiting.")
            break
        else:
            continue  # retry
            
    restaurant_review: RestaurantReview = parser.parse(content)

    print("\nParsed Restaurant Review Information:")

    print(restaurant_review.model_dump_json(indent=2))
    print("It was succcessful!")

    break  # exit the retry loop on success











