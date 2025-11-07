from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse


load_dotenv()  # take environment variables from .env.

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    temperature=0.5,
                                    max_tokens=2048)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are concise assistant.Answer as briefly as possible"),
        ("user", "{input}?")
    ]
)

chain = prompt | llm_gemini

print("stateless Demo. Type 'exit' to quit.") 
while True:
    user_input = input("Enter your question: ")
    if user_input.lower() == 'exit':
        print("Exiting the demo. Goodbye!")
        break

    variables = {"input": user_input}
    response = chain.invoke(variables)
    print(f"Bot:{response.content}\n")
    print(response.content)
