from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()  # take environment variables from .env.


llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate.from_template(
    """ Answer the following questions using the cotext below.
    If the questions are not in the context, say "I don't know" and don't make up an answer.
    Context: {context}
    Question: {question}
    Answer:"""
)
#lcel composition
chain = template | llm_gemini

response = chain.invoke({"context": "Capital city of France is Paris",
                         "question": "What is the capital city of France?"})

print("Response from Gemini:")
print(response.content)


