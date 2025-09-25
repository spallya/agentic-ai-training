from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

# Initialize the language model
language_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define tools

@tool
def find_attractions(city_name: str) -> str:
    """Fetch details about the top 3 tourist attractions in a city."""
    prompt = PromptTemplate.from_template(
        "List the top 3 tourist attractions in the city: {city_name}"
    )
    query = prompt.format(city_name=city_name)
    return language_model.invoke(query).content


@tool
def create_summary(details: str) -> str:
    """Convert detailed research into concise bullet points."""
    prompt = PromptTemplate.from_template(
        "Summarize the following details into bullet points:\n\n{details}"
    )
    query = prompt.format(details=details)
    return language_model.invoke(query).content


@tool
def plan_itinerary(details: str) -> str:
    """Create a 1-day travel itinerary based on the provided details."""
    prompt = PromptTemplate.from_template(
        "Based on this information, create a 1-day travel itinerary:\n\n{details}"
    )
    query = prompt.format(details=details)
    return language_model.invoke(query).content


# Define the pipeline
travel_pipeline = find_attractions | RunnableParallel({
    "summary": create_summary,
    "itinerary": plan_itinerary
})

if __name__ == "__main__":
    destination = "Kolkata"
    output = travel_pipeline.invoke(destination)

    print("\n--- Summary of Attractions ---\n", output["summary"])
    print("\n--- Suggested Itinerary ---\n", output["itinerary"])