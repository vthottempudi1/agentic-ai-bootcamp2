from typing import TypedDict, Optional,List, Annotated 
from langgraph.graph import  START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
from wikipedia import summary as wiki_summary
import os
#import numexpr


load_dotenv()


class WorkflowState(TypedDict, total=False):
    """
    query: user input query
    """
    query: str
    context: Annotated[str, 'text fetched from wikipedia']
    summary: Annotated[str, 'LLM-generated summary']
    summary_formatted: Annotated[str, 'formatted summary with bullet points']


@tool
def wikipedia_tool(topic: str) -> str:
    try:
        # Clean the query - remove command-like text and focus on the actual topic
        clean_topic = topic.replace("Write about", "").replace("write about", "").strip()
        if not clean_topic:
            clean_topic = "Artificial Intelligence"
        
        print(f"Searching Wikipedia for: '{clean_topic}'")
        result = wiki_summary(clean_topic, sentences=10, auto_suggest=True, redirect=True)
        print(f"Retrieved {len(result)} characters from Wikipedia")
        return result
    except Exception as e:
        # Try alternative search terms
        try:
            if "AI" in topic.upper():
                result = wiki_summary("Artificial Intelligence", sentences=10, auto_suggest=True, redirect=True)
                print(f"Retrieved {len(result)} characters from Wikipedia (fallback)")
                return result
        except:
            pass
        return f'Could not fetch from Wikipedia for {topic}. Error: {e}'



llm_gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash',
                                 google_api_key=os.getenv('GOOGLE_API_KEY'), 
                                 temperature=0.3)

@tool
def summarize_tool(text: str) -> str:
    if not text or 'Could not fetch' in text:
        return 'No valid context to summarize.'
    prompt = PromptTemplate.from_template('''You are help assistant that summarizes technical/contextual text.  Summarize the following text in 4–6 lines, keeping product names intact.
                                          {context} Summary:''')
    
    chain = prompt | llm_gemini
    out = chain.invoke({'context': text})
    return out.content


## 5.1 Research Agent (Wikipedia tool)
def research_agent(state: WorkflowState) -> WorkflowState:
    query = (state.get('query') or '').strip()
    if not query:
        state['context'] = 'No query provided.'
        return state

    print(f"🔍 Research Agent: Processing query '{query}'...")
    context = wikipedia_tool(query)
    state['context'] = context
    print(f"Research Agent: Context retrieved ({len(context)} characters)")
    return state

## 5.2 Summarization Agent( summarize_tool)
def summary_agent(state: WorkflowState) -> WorkflowState:
    context = state.get('context') or ''
    print(f"📝 Summary Agent: Processing context ({len(context)} characters)...")
    summary = summarize_tool(context)
    state['summary'] = summary
    print(f"✅ Summary Agent: Generated summary ({len(summary)} characters)")
    return state

#6. Build the workflow graph (Write it with Langgraph)
def build_workflow():
    graph= StateGraph(WorkflowState)
    graph.add_node("research_agent", research_agent)
    graph.add_node("summary_agent", summary_agent)
    graph.add_edge(START, "research_agent")
    graph.add_edge("research_agent", "summary_agent")
    graph.add_edge("summary_agent", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer)


@tool
def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Agent 3: Formats summary into bullet points"""
    query = state.get('query', 'Unknown Topic')
    summary = state.get('summary', '')
    
    print(f"Agent 3 (Formatter): Formatting summary for '{query}'...")
    
    # Creates format: "Summary for <query>: - Point 1 - Point 2 - Point 3"
    format_prompt = PromptTemplate.from_template('''Please format the following summary into exactly 3 bullet points for the topic "{query}":
                                                 {summary}
                                                 Format as:
                                                 Summary for {query}:
                                                 - Point 1
                                                 - Point 2  
                                                 - Point 3
                                                 Keep each point concise and informative.''')
    
    chain = format_prompt | llm_gemini
    formatted_result = chain.invoke({'query': query, 'summary': summary})
    formatted_summary = formatted_result.content.strip()
    
    # Add source attribution
    formatted_summary += "\n\nSource: Wikipedia"
    
    return {'summary_formatted': formatted_summary}  # ← Stores in state["summary_formatted"]



builder = StateGraph(WorkflowState)
builder.add_node("search_agent", research_agent)     # Agent 1: Wikipedia search
builder.add_node("summary_agent", summary_agent)     # Agent 2: Summarization  
builder.add_node("formatter_agent", formatter_agent) # Agent 3: Formatting

# Flow: START → search_agent → summary_agent → formatter_agent → END
builder.add_edge(START, "search_agent")
builder.add_edge("search_agent", "summary_agent") 
builder.add_edge("summary_agent", "formatter_agent")
builder.add_edge("formatter_agent", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="three-agents-workflow.png")
print("Workflow diagram saved as 'three-agents-workflow.png'")
state = WorkflowState()


if __name__ == "__main__":
    user_query = input("Enter a topic to summarize: ")
    state["query"] = user_query

    final_state = graph.invoke(state)
    print("\n Final Output:\n")
    print(final_state["summary_formatted"])


## python 19.two-agents-langgraph.py












