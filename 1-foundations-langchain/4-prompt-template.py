from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

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

template = PromptTemplate.from_template(
    """ Answer the following questions using the cotext below.
    If the questions are not in the context, say "I don't know" and don't make up an answer.
    Context: {context}
    Question: {question}
    Answer:"""
)
template_object = template.invoke(
    {"context": "Capital city of France is Paris",
     "question": "What is the capital city of France?"}
)

 
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")



response_openai = llm_openai.invoke(template_object)
response_gemini = llm_gemini.invoke(template_object)

print("Response from OpenAI:")
print(response_openai.content)

print("Response from Gemini:")
print(response_gemini.content)
