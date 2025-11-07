import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from wikipedia import summary as wiki_summary
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition


#1.Load Environment(here we are loading environment variables for Google API key)

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

#2. You will build this flow:START → research_agent (calls wikipedia_tool) → summary_agent (calls summarize_with_gemini_tool) → END

#3.Define shared state
class WorkflowState(TypedDict, total=False):
    query: str
    context: Annotated[str, 'text fetched from wikipedia']
    summary: Annotated[str, 'LLM-generated summary']
    summary_formatted: Annotated[str, 'formatted summary with bullet points']

#4.Define Tools
#4.1 Wikipedia tool
@tool
def wikipedia_tool(topic: str) -> str:
    """Fetch context from wikipedia"""
    try:
        # Clean the query - remove phrases like "write about", "tell me about", etc.
        clean_topic = topic.replace("write about", "").replace("tell me about", "").replace("Write about", "").replace("Tell me about", "").strip()
        if not clean_topic:
            clean_topic = topic.strip()
        
        print(f"Searching Wikipedia for: '{clean_topic}'")
        return wiki_summary(clean_topic, sentences=8, auto_suggest=True, redirect=True)
    except Exception as e:
        return f"Error fetching data from Wikipedia for {topic}. Error: {e}"
search_tool = DuckDuckGoSearchRun()
tools = [search_tool, wikipedia_tool]   

# Gemini model (LLM)

gemini = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

#4.2 Summarize tool
@tool
def summarize_tool(text: str) -> str:
    """Summarize text using Gemini LLM"""
    if not text or 'Could not fetch' in text:
        return 'No valid context to summarize.'
    prompt = PromptTemplate.from_template('''You are a helpful assistant that summarizes technical/contextual text. Summarize the following the text in 4–6 lines, keeping product names intact.
                                          {context}
                                          Summary:''')
    chain = prompt | gemini
    out = chain.invoke({'context': text})
    return out.content
#search_tool = DuckDuckGoSearchRun()
tools = [wikipedia_tool, summarize_tool]

#5. Define Agents
#5.1 Reserch Agent

def research_agent(state: WorkflowState) -> WorkflowState:
    """Agent 1: Fetches context from Wikipedia"""
    query = (state.get('query') or '').strip()
    if not query:
        state['context'] = 'No query provided.'
        return state
    
    print(f"Research Agent: Fetching Wikipedia content for '{query}'...")
    # Call the tool function directly (invoke method for StructuredTool)
    context = wikipedia_tool.invoke({"topic": query})
    state['context'] = context
    print(f"Research Agent: Retrieved {len(context)} characters")
    return state

#5.2 Summary Agent
def summary_agent(state: WorkflowState) -> WorkflowState:
    """Agent 2: Summarizes the fetched context"""
    context = state.get('context') or ''
    print(f"Summary Agent: Processing {len(context)} characters...")
    # Call the tool function directly (invoke method for StructuredTool)
    summary = summarize_tool.invoke({"text": context})
    state['summary'] = summary
    print(f"Summary Agent: Generated {len(summary)} characters summary")
    return state

def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Agent 3: Formats summary into 3 bullet points"""
    query = state.get('query', 'Unknown Topic')
    summary = state.get('summary', '')
    
    if not summary or summary == 'No valid context to summarize.':
        state['summary_formatted'] = f'Summary for {query}:\n- No summary available\n\nSource: Wikipedia'
        return state
    
    print(f"Formatter Agent: Creating bullet points for '{query}'...")
    
    prompt = PromptTemplate.from_template('''You are a helpful assistant that formats text into exactly 3 concise bullet points.

Summary: {summary}

Format as:
Summary for {query}:
- Point 1
- Point 2
- Point 3

Source: Wikipedia

Keep each point informative and concise.''')
    
    chain = prompt | gemini
    formatted_output = chain.invoke({"summary": summary, "query": query})
    state["summary_formatted"] = formatted_output.content.strip()
    print(f"Formatter Agent: Bullet point summary created")
    return state

# 6 Write with Langgraph (Workflow graph)
def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node('research_agent', research_agent)
    graph.add_node('summary_agent', summary_agent)
    graph.add_node('formatter_agent', formatter_agent)  # Add formatter agent to graph
    graph.add_edge(START, 'research_agent')
    graph.add_edge('research_agent', 'summary_agent')
    graph.add_edge('summary_agent', 'formatter_agent')
    graph.add_edge('formatter_agent', END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

#6.2 Run the workflow

if __name__ == "__main__":
    user_query = input("Enter a topic to summarize: ").strip()

    if not user_query:
        print("No topic entered. Exiting.")
        exit()

    workflow = build_workflow()
    initial_state = {"query": user_query}

    final_state = workflow.invoke(
        initial_state, config={"configurable": {"thread_id": "demo_user_1"}}
    )

    print('Query:', initial_state['query'])
    print('\nContext:\n', final_state.get('context'))
    print('\nSummary:\n', final_state.get('summary'))
    print('\nFormatted Summary:\n', final_state.get('summary_formatted'))






