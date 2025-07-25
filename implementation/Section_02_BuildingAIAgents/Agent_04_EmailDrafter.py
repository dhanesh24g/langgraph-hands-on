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
    """Saves the document with a suitable Text filename and finishes processing.

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

