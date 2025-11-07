import argparse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # take environment variables from .env.


args = argparse.ArgumentParser(description="Dynamic Prompt Template with Argparse")
args.add_argument("--city", type=str, required=True, help="City for travel recommendations")
args.add_argument("--days", type=str, required=True, help="Number of days for the trip")
args.add_argument("--budget", type=str, required=True, help="Budget for the trip")
args.add_argument("--traveler_type", type=str, required=True, help="Type of traveler")
args.add_argument("--length", type=str, required=True, help="Length of the recommendation")
parsed_args = args.parse_args()

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    temperature=0.5,
                                    max_tokens=2014)



       
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert travel assistant who give clear, engaging, and realistic travel recommendations."
        "Adapt explanations to the user's audience and style."
        "Use short sentences and concrete examples. Be friendly and professional."),
        ("user",
         """Provide a {days} travel recommendation for {traveler_type} about visiting {city} for a {budget}.
         Keep it {length}.
         Format: \n
         Opener: 1-2 lines introducing about city \n
         Itinerary: Day-by-day plan matching the days \n
         Travel Tips: 1-2 short tips (best time to go, local customs, or cost insights) \n
         Bottom line: Single sentence starting with 'Bottom line:' summarizing why they should visit.
         """
         )
    ]
)

chain = prompt | llm_gemini 

variables = {
    "city": parsed_args.city,
    "days": parsed_args.days,
    "budget": parsed_args.budget,
    "traveler_type": parsed_args.traveler_type,
    "length": parsed_args.length
}

response = chain.invoke(variables)
print("Response from Gemini via LCEL:")
print(response.content)


#python 7-dynamic-prompt-template-argparse.py --topic "Agentic AI" --audience "product manager" --style "executive brief" --length "under 120 words"
##python 7-dynamic-prompt-template-argparse.py --topic "Agentic AI" --audience "generative ai engineer" --style "dev intense brief" --length "150 words"


#9-dynamic-prompt-template-argparse-TravelRecommenderAgent.py
#python 9-dynamic-prompt-template-argparse-TravelRecommenderAgent.py --city "Goa" --days "3 days" --budget "moderate" --traveler_type "family" --length "150"