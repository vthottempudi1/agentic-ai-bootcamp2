from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # take environment variables from .env.

# https://python.langchain.com/docs/integrations/chat/openai/
# https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI


llm_openai = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# llm_openai_V2 = ChatOpenAI(
#     model="gpt-5-turbo",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# ) 
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
messages = [
    (
        "system",
        "You are a helpful funny assistant that answers the questions in a humorous way. Translate the user sentence.",
    ),
    ("human", "Who is the President of United States?"),
]
ai_msg = llm_openai.invoke(messages)
ai_msg = llm_gemini.invoke(messages)

response_openai = llm_openai.invoke(messages)
response_gemini = llm_gemini.invoke(messages)

print("Response from OpenAI:")
print(response_openai.content)

print("Response from Gemini:")
print(response_gemini.content)
