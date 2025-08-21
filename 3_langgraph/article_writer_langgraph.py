from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define State schema
class ArticleState(TypedDict, total=False):
    topic: str
    research: str
    outline: str
    article: str

# Define Agents
def research_agent(state: ArticleState) -> ArticleState:
    query = state["topic"]
    print("Researching on topic: ", query)
    research_output = llm.invoke(f"Research about: {query}").content
    state["research"] = research_output
    return state

def outline_agent(state: ArticleState) -> ArticleState:
    research = state["research"]
    print("Generating Outline on topic: ", state["topic"])
    outline = llm.invoke(f"Create a detailed outline based on this research: {research}").content
    state["outline"] = outline
    return state

def writer_agent(state: ArticleState) -> ArticleState:
    outline = state["outline"]
    print("Generating Article on topic: ", state["topic"])
    article = llm.invoke(f"Write a detailed article from this outline: {outline}").content
    state["article"] = article
    return state

# Build LangGraph
builder = StateGraph(ArticleState)
builder.add_node("research", research_agent)
builder.add_node("outline", outline_agent)
builder.add_node("writer", writer_agent)

builder.add_edge(START, "research")
builder.add_edge("research", "outline")
builder.add_edge("outline", "writer")
builder.add_edge("writer", END)

# Compile graph
graph = builder.compile()
# graph.get_graph().draw_png("graphs/article_writer_langgraph.png")
print(graph.get_graph().print_ascii())
# Run
initial_state = ArticleState(topic="Impact of AI in Education")
result = graph.invoke(initial_state)
print(result["article"])