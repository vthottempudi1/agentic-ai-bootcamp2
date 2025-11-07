from typing import TypedDict, Optional, List
from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import ast
import numexpr


load_dotenv()

@tool
def calculator(query: str) -> str:
    """A simple calculator that evaluates mathematical expressions."""
    try:
        # Safely evaluate the expression using ast.literal_eval for safety
        result = eval(query)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


search_tool = DuckDuckGoSearchRun()
tools = [search_tool, calculator]

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048).bind_tools(tools)

class State(TypedDict):
    """
    messages: user input messages
    """
    messages: str    
    response: Optional[str]
    

def llm_node(state: State) -> State:
    result = llm_gemini.invoke(state["messages"])
    return {"response": result.content}

builder = StateGraph(State)
builder.add_node("model", llm_node)
builder.add_edge("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edge("model", tools_condition)
builder.add_edge("tools", "model")
builder.add_edge("model", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="basic-agent1.png")

input = {
    "messages": "calculate 123*45."
}

result = graph.invoke(input)
print("Final Result:")
print(result["response"])


