from os import write
from os.path import exists
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model='gpt-4o-mini')

file_store = 'Agent_02_ChatLog.txt'

def process_message(state: AgentState) -> AgentState:
    """Takes input and returns the AI output token-by-token"""

    response = llm.invoke(state["messages"])
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

    print("\nState: ", state)
    return state


graph = StateGraph(AgentState)

graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []
if exists(file_store):
    with open(file_store, 'r') as fs:
        conversation_history = fs.readlines()

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    print(result['messages'])
    conversation_history = result['messages']

    user_input = input("\nEnter: ")


with open(file_store, "a") as file:
    file.write("\nYour conversation history\n")
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"Human: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n\n")
    file.write("End of conversation\n")

print("Conversation saved to Agent_02_ChatLog.txt")
