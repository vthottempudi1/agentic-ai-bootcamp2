from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationChain
import argparse


load_dotenv()  # take environment variables from .env.

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    temperature=0.5,
                                    max_tokens=2048)

# Simple memory implementation using a list to store conversation history
conversation_history = []

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant. Answer as briefly as possible. Use the conversation history for context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

chain = prompt | llm_gemini

print("Memory Demo with Conversation History. Type 'exit' to quit.")
print("Type 'show' to print the conversation history.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Exiting the demo. Goodbye!")
        break
    elif user_input.lower() == 'show':
        print("Conversation History:")
        for i, message in enumerate(conversation_history):
            if isinstance(message, HumanMessage):
                print(f"{i+1}. Human: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"{i+1}. AI: {message.content}")
        print("-----------------------------------\n")
        continue

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Get response from chain with conversation history
    response = chain.invoke({
        "input": user_input,
        "history": conversation_history
    })
    
    # Add AI response to history
    conversation_history.append(AIMessage(content=response.content))
    
    print(f"Bot: {response.content}\n")
    
