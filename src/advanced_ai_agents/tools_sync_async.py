from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from pydantic import BaseModel
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
import gradio as gr

load_dotenv(override=True)

# Building Two Finance Specific tools
class CurrencyArgs(BaseModel):
    from_currency: str
    to_currency: str

alpha_vantage = AlphaVantageAPIWrapper()

currency_tool = StructuredTool.from_function(
    name="currency_conversion_tool",
    func=alpha_vantage.run,
    args_schema=CurrencyArgs,
    description=
    """
    This tool gets the currency exchange rates for the given currency pair.
    This method mandates 2 Currency Codes as arguments: from_currency, to_currency.

    Args:
        from_currency: The first currency mentioned in the query.
        to_currency: The second currency mentioned in the query.
    """
)

stock_tool = Tool(
    name="stock_tool",
    func=alpha_vantage._get_quote_endpoint,
    description=
    """
    This tool gets the latest stock exchange rates from the USA Stock market for the given stock.

    Args:
        stock_code = US Stock Exchange code of the stock you want to inquire.
    """
)

# Adding the second tool (PlayWright)
async_browser = create_async_playwright_browser(headless=False)
browser = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
browser_tools = browser.get_tools()

tools = []
tools = browser_tools
tools.append(currency_tool)
tools.append(stock_tool)

llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# Building the Graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(AgentState)

# Building the Chatbot for user
def chatbot(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    "tools"
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "3"}}

async def chat(user_input: str, history):
    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    return response["messages"][-1].content

gr.ChatInterface(chat, type="messages").launch()