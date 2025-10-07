import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is missing. Please check your .env file.")

# Define tools
@tool
def fetch_weather(city_name: str) -> str:
    """Retrieve the current weather for a specified city."""
    weather_data = {
        "Mumbai": "Humid 32° C",
        "Tokyo": "Rainy 22° C",
        "New York": "Cloudy 15° C",
    }
    return weather_data.get(city_name, "Clear 27° C")


@tool
def find_attractions(city_name: str) -> str:
    """Fetch the top tourist attractions for a specified city."""
    attractions_data = {
        "Mumbai": ["Gateway of India", "Marine Drive", "Elephanta Caves"],
        "Tokyo": ["Tokyo Tower", "Shinjuku Gyoen", "Meiji Shrine"],
        "New York": ["Statue of Liberty", "Central Park", "Times Square"],
    }
    return ", ".join(attractions_data.get(city_name, ["City Center", "Local Museum"]))


# Collect tools
available_tools = [fetch_weather, find_attractions]

# Initialize the language model
language_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize the agent
travel_agent = initialize_agent(
    tools=available_tools,
    llm=language_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Run the query
user_query = "You are my trip planner, plan a day for me in Tokyo"
agent_response = travel_agent.run(user_query)

# Display the result
print("\n--- Final Plan ---")
print(agent_response)