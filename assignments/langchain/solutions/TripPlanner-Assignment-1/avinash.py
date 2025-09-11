from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#tools
@tool
def research_tool(city: str) -> str:
    """Research 3 top tourist attractions from a given city."""
    print("Researching 3 top tourist attractions for city:", city)
    return llm.invoke(f"Research 3 top tourist attractions from a given city: {city}").content


@tool
def summarize_tool(research: str) -> str:
    """Summarize the research text into key points."""
    print("Summarizing Research...")
    return llm.invoke(f"Summarize this research into key bullet points: {research}").content


@tool
def itinerary_tool(itinerary_text: str) -> str:
    """Generate a short 1-day itinerary covering those top 3 attractions."""
    return llm.invoke(f"Generate a short 1-day itinerary covering those top 3 attractions: {itinerary_text}").content



#Pipeline
pipeline = research_tool | RunnableParallel({
    "summary": summarize_tool,
    "itinerary": itinerary_tool
})

if __name__ == "__main__":
    city = "Bangalore"
    result = pipeline.invoke(city)

    print("**** Summary ****:\n", result["summary"])
    print("\n**** Itinerary ****\n", result["itinerary"])