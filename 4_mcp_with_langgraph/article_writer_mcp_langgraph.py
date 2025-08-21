from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from math_mcp_server.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Init MCP server
mcp = FastMCP("Article Generator")

# Async research tool
@mcp.tool()
async def research_tool(topic: str) -> str:
    print(f"[MCP] Researching topic: {topic}")
    result = (await llm.invoke(f"Research about: {topic}")).content
    return result

# Research agent calls the tool directly
def research_agent(state: dict) -> dict:
    topic = state["topic"]
    print("Researching topic via MCP tool:", topic)
    import asyncio
    result = asyncio.run(research_tool(topic))
    state["research"] = result
    return state

# Outline agent
def outline_agent(state: dict) -> dict:
    research = state["research"]
    print("Generating Outline on topic:", state["topic"])
    outline = llm.invoke(f"Create a detailed outline based on this research: {research}").content
    state["outline"] = outline
    return state

# Writer agent
def writer_agent(state: dict) -> dict:
    outline = state["outline"]
    print("Generating Article on topic:", state["topic"])
    article = llm.invoke(f"Write a detailed article from this outline: {outline}").content
    state["article"] = article
    return state

# State schema
class ArticleState(TypedDict, total=False):
    topic: str
    research: str
    outline: str
    article: str

# Build LangGraph
builder = StateGraph(ArticleState)
builder.add_node("research", research_agent)
builder.add_node("outline", outline_agent)
builder.add_node("writer", writer_agent)
builder.add_edge(START, "research")
builder.add_edge("research", "outline")
builder.add_edge("outline", "writer")
builder.add_edge("writer", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = ArticleState(topic="Impact of AI in Education")
    final_state = graph.invoke(initial_state)
    print("===== Generated Article =====")
    print(final_state["article"])