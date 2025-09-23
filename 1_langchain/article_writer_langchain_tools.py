from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---- Define Tools ----
@tool
def research_tool(topic: str) -> str:
    """Research a given topic and return findings."""
    print("Researching on topic:", topic)
    return llm.invoke(f"Research about: {topic}").content


@tool
def outline_tool(research: str) -> str:
    """Generate a detailed outline based on research text."""
    print("Generating Outline...")
    return llm.invoke(f"Create a detailed outline based on this research: {research}").content


@tool
def writer_tool(outline: str) -> str:
    """Generate a full article based on outline."""
    print("Generating Article...")
    return llm.invoke(f"Write a detailed article from this outline: {outline}").content


@tool
def summary_tool(research: str) -> str:
    """Summarize the research text into key points."""
    print("Summarizing Research...")
    return llm.invoke(f"Summarize this research into key bullet points: {research}").content


# ---- Build Pipeline with Parallel Branches ----
pipeline = RunnableSequence(
    {
        # Run research first
        "research": research_tool,
    }
    | RunnableParallel(
        {
            # Branch 1: Outline -> Article
            "article": RunnableSequence(
                lambda x: x["research"],  # extract research
                outline_tool,
                writer_tool,
            ),
            # Branch 2: Summary directly from research
            "summary": RunnableSequence(
                lambda x: x["research"],
                summary_tool,
            ),
        }
    )
)


# ---- Run ----
if __name__ == "__main__":
    topic = "Impact of AI in Education"
    result = pipeline.invoke(topic)

    print("\n\n=== Results ===\n")
    print("Summary:\n", result["summary"])
    print("\nArticle:\n", result["article"])