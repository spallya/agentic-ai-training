from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load .env variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found for calling llm")

# LLM setup with llama-3.1-8b-instant model from Groq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7) 

# Chain of tools
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


# pipeline
pipeline = research_tool | RunnableParallel({
    "summary": summarize_tool,
    "itinerary": itinerary_tool
})

if __name__ == "__main__":
    city = input("Enter a destination city: ")
    result = pipeline.invoke(city)

    print("\n *** About City *** \n", result["summary"])
    print("\n *** Your plan *** \n", result["itinerary"])

