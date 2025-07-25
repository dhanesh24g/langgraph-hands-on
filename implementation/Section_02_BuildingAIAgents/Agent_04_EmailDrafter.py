from os import system
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# Global variable to store the document content (Alternate to InjectedState)
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

#------------ Tools Section Start ------------#

@tool
def update(content: str) -> str:
    """Updates the document with provided content."""
    global document_content
    document_content = content

    return f"The document has been updated with the following content:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Saves the document with a suitable Text filename, starting with 'Agent_04_' and finishes processing.

    Args:
        filename: Name for the Text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            print(f"\n💾Document has been saved successfully to {filename}.")

            return f"Document has been saved successfully to {filename}."

    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"

#------------ Tools Section End ------------#

toolkit = [update, save]

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools=toolkit)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        
        - If the user wants to update or modify content, use 'update' tool with the complete updated content.
        - If the user wants to save and finish, use 'save' tool with the complete saved content.
        - Make sure to always show the current document state after modifications.
        
        The current document content is: {document_content}
        """)

    if not state["messages"]:
        user_input = input(f"I am ready to help you update a document. What would you like to create?\n")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?\n")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"⚙️ USING TOOLS {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        "continue"

    # Look for the most recent tool message and decide
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"

    return "continue"

def print_messages(messages):
    """Function created to print the messages in a more readable format."""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

#------------ Build and Compile Graph Start ------------#

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(toolkit))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

#------------ Build and Compile Graph End ------------#

def run_document_agent():

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])


if __name__ == "__main__":
    run_document_agent()