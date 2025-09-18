from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load env variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found for calling llm")

# LLM setup
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


# Tools

@tool
def research_tool(city: str) -> str:
    """Research 3 top tourist attractions from a given city."""
    prompt = PromptTemplate.from_template(
        "Research 3 top tourist attractions from the city: {city}"
    )
    formatted_prompt = prompt.format(city=city)
    return llm.invoke(formatted_prompt).content


@tool
def summarize_tool(research: str) -> str:
    """Summarize research into concise bullet points."""
    prompt = PromptTemplate.from_template(
        "Summarize the following research into bullet points:\n\n{research}"
    )
    formatted_prompt = prompt.format(research=research)
    return llm.invoke(formatted_prompt).content


@tool
def itinerary_tool(research: str) -> str:
    """Generate a short 1-day travel itinerary using the research."""
    prompt = PromptTemplate.from_template(
        "Using this research, generate a short 1-day travel itinerary:\n\n{research}"
    )
    formatted_prompt = prompt.format(research=research)
    return llm.invoke(formatted_prompt).content


# Pipeline
pipeline = research_tool | RunnableParallel({
    "summary": summarize_tool,
    "itinerary": itinerary_tool
})

if __name__ == "__main__":
    city = "Hyderabad"
    result = pipeline.invoke(city)

    print("\n--- Summary ---\n", result["summary"])
    print("\n--- Itinerary ---\n", result["itinerary"])

