from typing import TypedDict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

#take environment variables fro .env.

load_dotenv()

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                    temperature=0.5)


def llm_node(state: dict) -> dict:
    user_input = state.get("input", "")
    #from langchain_core.messages import HumanMessage
    #response = llm_gemini.invoke([HumanMessage(content=user_input)])
    response = llm_gemini.invoke(user_input)
    return {"Output": response.content}


class GraphState(TypedDict):
    input: str
    Output: str

graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)


#input_state = {"input": "Explain LangGraph in one sentence."}
input_state = {"prompt": "Explain LangGraph in one sentence.","response":"placeholder"}
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-1234"}}

print("Running LangGraph...")
result = app.invoke(input_state, config=config)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Final Output:", result["Output"])


snapshot = checkpointer.get_state(config)
print("Snapshot Values:")
print(snapshot.values)


#python 14.hello-langgraph.py


# # Generate graph visualization
# print("\nGenerating graph visualization...")

# try:
#     # Method 1: Save as PNG (requires graphviz)
#     graph_image = app.get_graph().draw_mermaid_png()
#     with open("langgraph_diagram.png", "wb") as f:
#         f.write(graph_image)
#     print("✅ Graph saved as 'langgraph_diagram.png'")
# except Exception as e:
#     print(f"❌ PNG generation failed: {e}")

# try:
#     # Method 2: Print Mermaid text (always works)
#     mermaid_code = app.get_graph().draw_mermaid()
#     print("\n📊 Mermaid Graph Code:")
#     print(mermaid_code)
    
#     # Save mermaid code to file
#     with open("langgraph_diagram.mmd", "w") as f:
#         f.write(mermaid_code)
#     print("✅ Mermaid code saved as 'langgraph_diagram.mmd'")
# except Exception as e:
#     print(f"❌ Mermaid generation failed: {e}")

# try:
#     # Method 3: ASCII representation
#     ascii_graph = app.get_graph().draw_ascii()
#     print("\n🎨 ASCII Graph:")
#     print(ascii_graph)
# except Exception as e:
#     print(f"❌ ASCII generation failed: {e}")



