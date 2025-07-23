from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

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


def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=list_of_tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input_prompt = {"messages": "I have 20 dollars in portfolio and I received 30 dollars in profit. Also, I received 20% appreciation on the total value. How much my account value?"}

print_stream(app.stream(input=input_prompt, stream_mode="values"))

