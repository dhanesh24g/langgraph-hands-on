from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """This tool returns the sum of 2 numbers"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This tool returns the subtraction of 2 numbers"""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """This tool returns the multiplication of 2 numbers"""
    return a * b

list_of_tools = [add, subtract, multiply]

llm = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools=list_of_tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant. Please answer my query to the best of your knowledge."
    )

    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.add_node("tool_node", ToolNode(tools=list_of_tools))

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)

graph.add_edge("tool_node", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input_prompt = {"messages": "I have 200 dollars in portfolio and I received 10 dollars in dividend. Also, I received 20% appreciation on the portfolio value. How much is my account value? Also who is Apple CEO ?"}

print_stream(app.stream(input=input_prompt, stream_mode="values"))