import uuid
import yaml
from dotenv import load_dotenv
from typing import Literal
import os

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

from agent_node_factory import *

from tools import (
    load_csv_file,
    get_user_input_tool,
    execute_sql_query,
    get_database_schema,
)

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Read prompts.yaml file and extract agent prompts
with open('prompts.yaml', 'r', encoding='utf-8') as file:
    prompt_data = yaml.safe_load(file)
execution_agent_prompt = prompt_data['execution_agent']['system_message']

# Create ChatPromptTemplate instance
execution_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", execution_agent_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#########
#invoke GPT 4o
#########
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    api_key=openai_api_key,
    max_retries=3,      
    request_timeout=20, 
)



# Define tool list
execution_tools = [
    get_user_input_tool,
    execute_sql_query,
    get_database_schema,
]

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        return {"messages": [result]}
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result]}


# Router function to guide the workflow based on conditions
def execution_router(state) -> Literal["DECISION ANALYSIS", "wait_for_input", "call_tool"]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "call_tool"
    if "DECISION ANALYSIS" in last_message.content:
        return "DECISION ANALYSIS"
    if "COMPLETE TASK" in last_message.content:
        return "wait_for_input"
    
    return "DECISION ANALYSIS"

# Node function to get user input
# def get_user_input_node(state) -> dict:
#     """Node function to get user input and add it to the state"""
#     try:
#         user_input_result = get_user_input_tool()
#         if "error" in user_input_result:
#             return {"messages": [HumanMessage(content="Let's continue our conversation.")]}
        
#         return {"messages": user_input_result["messages"]}
#     except Exception as e:
#         return {"messages": [HumanMessage(content=f"Error getting input: {str(e)}")]}
def get_user_input_node(state) -> dict:
    """Node function to get user input and add it to the state"""
    try:
        
        user_input_result = get_user_input_tool.invoke({})
        
        if isinstance(user_input_result, dict) and "error" in user_input_result:
            return {"messages": [HumanMessage(content="Let's continue our conversation.")]}
        
        if isinstance(user_input_result, dict) and "messages" in user_input_result:
            return {"messages": user_input_result["messages"]}
        
        
        return {"messages": [HumanMessage(content=str(user_input_result))]}
    except Exception as e:
        print(f"Error in get_user_input_node: {str(e)}")
        
        try:
            user_input = input("\n(Error occurred. Please try again): ")
            return {"messages": [HumanMessage(content=user_input)]}
        except:
            return {"messages": [HumanMessage(content="Please continue.")]}
# Create execution tool node with fallback mechanism
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls 

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Define the workflow
workflow = StateGraph(State)

execution_assistant_runnable = execution_assistant_prompt | llm.bind_tools(execution_tools)

# Add nodes to the workflow
workflow.add_node("Execution_Assistant", Assistant(execution_assistant_runnable))
workflow.add_node("Execution_Tools", create_tool_node_with_fallback(execution_tools))
workflow.add_node("wait_for_input", get_user_input_node)

# Modify the edge between ToolNode and Execution_Assistant
workflow.add_edge("Execution_Tools", "Execution_Assistant")
workflow.add_edge("wait_for_input", "Execution_Assistant")

# Add conditional edges for routing
workflow.add_conditional_edges(
    "Execution_Assistant",
    execution_router,
    {
        "DECISION ANALYSIS": "Execution_Assistant",
        "call_tool": "Execution_Tools",  
        "wait_for_input": "wait_for_input", 
    },
)

workflow.add_edge(START, "Execution_Assistant")

# Set up memory saver
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Generate and save Mermaid image as a PNG file
'''
graph.get_graph().draw_mermaid_png(
    curve_style=CurveStyle.LINEAR,
    wrap_label_n_words=9,
    output_file_path='generated_files/agent_graph.png',
    draw_method=MermaidDrawMethod.API,
    background_color="white",
    padding=10,
)
'''

