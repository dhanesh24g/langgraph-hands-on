from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.1:8b",
    temperature=0.6
)

class AgentState(TypedDict):
    messages: List[HumanMessage]


# Printing AI output iteratively (Token-wise) using flush and end params
def process_message(state: AgentState) -> AgentState:
    """Takes input and returns the AI output token-by-token"""
    response = llm.stream(state["messages"])
    print(f"AI: ", end="")
    for message in response:
        print(f"{message.content}", end="", flush=True)
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("\nEnter: ")
