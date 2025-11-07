# Two-Agent LangGraph Workflow with Tools
# Research Agent (Wikipedia) + Summarizer Agent (Gemini LLM)

# 1. Prerequisites - Import required packages
from typing import Annotated, TypedDict
from wikipedia import summary as wiki_summary
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os

# Load environment variables
load_dotenv()

# 3. State Definition - Shared state for both agents
class WorkflowState(TypedDict, total=False):
    query: str
    context: Annotated[str, 'text fetched from wikipedia']
    summary: Annotated[str, 'LLM-generated summary']
    summary_formatted: Annotated[str, 'formatted summary with bullet points']

# 4. Define Your Tools
# 4.1 Wikipedia Tool
def wikipedia_tool(topic: str) -> str:
    try:
        return wiki_summary(topic, sentences=8, auto_suggest=False, redirect=True)
    except Exception as e:
        return f'Could not fetch from Wikipedia for {topic}. Error: {e}'

# 4.2 Gemini Summarizer Tool
gemini = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', 
    google_api_key=os.getenv('GOOGLE_API_KEY'), 
    temperature=0.3
)

def summarize_tool(text: str) -> str:
    if not text or 'Could not fetch' in text:
        return 'No valid context to summarize.'
    prompt = PromptTemplate.from_template('''You are a helpful assistant that summarizes technical/contextual text.
Summarize the following text in 4–6 lines, keeping product names intact.

{context}

Summary:''')
    chain = prompt | gemini
    out = chain.invoke({'context': text})
    return out.content

# 5. Build the Agents (Nodes)
# 5.1 Research Agent (uses wikipedia_tool)
def research_agent(state: WorkflowState) -> WorkflowState:
    query = (state.get('query') or '').strip()
    if not query:
        state['context'] = 'No query provided.'
        return state
    
    print(f"🔍 Research Agent: Fetching Wikipedia content for '{query}'...")
    context = wikipedia_tool(query)
    state['context'] = context
    print(f"📖 Research Agent: Retrieved {len(context)} characters")
    return state

# 5.2 Summary Agent (uses summarize_tool)
def summary_agent(state: WorkflowState) -> WorkflowState:
    context = state.get('context') or ''
    print(f"📝 Summary Agent: Processing {len(context)} characters...")
    summary = summarize_tool(context)
    state['summary'] = summary
    print(f"✅ Summary Agent: Generated {len(summary)} characters summary")
    return state

# 7A – Add a Formatter Agent
def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Agent 3: Formats summary into bullet points"""
    query = state.get('query', 'Unknown Topic')
    summary = state.get('summary', '')
    
    print(f"🎨 Formatter Agent: Creating bullet points for '{query}'...")
    
    if not summary or summary == 'No valid context to summarize.':
        formatted_summary = f"Summary for {query}:\n- No summary available"
    else:
        # Create a prompt to format the summary into bullet points
        format_prompt = PromptTemplate.from_template('''Please format the following summary into exactly 3 bullet points for the topic "{query}":

{summary}

Format as:
Summary for {query}:
- Point 1
- Point 2  
- Point 3

Keep each point concise and informative.''')
        
        chain = format_prompt | gemini
        formatted_result = chain.invoke({'query': query, 'summary': summary})
        formatted_summary = formatted_result.content.strip()
    
    # 7C – Add Source Attribution
    formatted_summary += "\n\nSource: Wikipedia"
    
    state['summary_formatted'] = formatted_summary
    print(f"🎨 Formatter Agent: Bullet point summary created")
    return state

# 6. Wire It with LangGraph
def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node('research_agent', research_agent)
    graph.add_node('summary_agent', summary_agent)
    graph.add_node('formatter_agent', formatter_agent)  # 7A: Added formatter agent
    
    # 7A: Updated graph flow - START → research_agent → summary_agent → formatter_agent → END
    graph.add_edge(START, 'research_agent')
    graph.add_edge('research_agent', 'summary_agent')
    graph.add_edge('summary_agent', 'formatter_agent')
    graph.add_edge('formatter_agent', END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# Main execution with all extensions
if __name__ == '__main__':
    print("🚀 Starting Three-Agent Wikipedia Summarizer...")
    print("=" * 60)
    
    # 7B – Make It Interactive: Ask user for query dynamically
    user_query = input("Enter a topic to summarize: ")
    
    if not user_query.strip():
        print("❌ No query provided. Exiting...")
        exit()
    
    workflow = build_workflow()
    
    # Feed user input into initial workflow state
    initial_state = {'query': user_query.strip()}
    
    print(f"\n🎯 Processing query: '{user_query}'")
    print("-" * 40)
    
    # Run the complete workflow
    final_state = workflow.invoke(
        initial_state, 
        config={'configurable': {'thread_id': 'demo_user_1'}}
    )
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("🎉 FINAL RESULTS:")
    print("=" * 60)
    print(f"📋 Query: {final_state.get('query', 'N/A')}")
    print(f"📖 Context Length: {len(final_state.get('context', ''))} characters")
    print(f"\n📝 Raw Summary:")
    print(final_state.get('summary', 'No summary generated'))
    print(f"\n🎨 Formatted Summary:")
    print(final_state.get('summary_formatted', 'No formatted summary generated'))
    print("=" * 60)

# Example usage:
# python 1.three-agents-langgraph.py


#7A Formatter Agent
def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Agent 3: Formats summary into bullet points"""
    summary = state.get('summary') or ''
    if not summary:
        state['summary_formatted'] = 'No summary to format.'
        return state
    
    prompt = PromptTemplate.from_template('''You are a helpful assistant that formats text into 3 concise bullet points.
                                          summary: {summary}
                                          Outputformat:
                                          Summary for {query}:
                                          - Point 1
                                          - Point 2
                                          - Point 3
                                          Append: 'Source: Wikipedia' at the end.''')
    chain = prompt | gemini
    formatted_output = chain.invoke({"summary": summary, "query": query})
    state["summary_formatted"] = formatted_output.content.strip()
    return state
    # chain = prompt | gemini
    # formatted_summary = chain.invoke({'summary': summary, 'query': state.get('query', '')})
    # state['summary_formatted'] = formatted_summary['text'].content
    # return state

#7B Make It Interactive (User input already handled in main execution)


# 7C Add Source Attribution (Handled in formatter_agent)                                        

