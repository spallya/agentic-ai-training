from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Mock data for attractions
ATTRACTIONS_DB: Dict[str, Dict[str, list]] = {
    "Delhi": {
        "outdoor": ["Red Fort - Historic monument", "Lodhi Gardens - Beautiful park"],
        "indoor": ["National Museum", "Akshardham Temple"]
    },
    "Paris": {
        "outdoor": ["Eiffel Tower", "Luxembourg Gardens"],
        "indoor": ["Louvre Museum", "Notre-Dame Cathedral"]
    }
}

# Tool 1: Weather Information
@tool
def get_weather(city: str) -> str:
    """Get weather for a specific city (mocked data)."""
    weather_data = {
        "Delhi": "Sunny 28°C",
        "Paris": "Rainy 20°C"
    }
    return weather_data.get(city, "Weather data not available")

# Tool 2: Search Attractions
@tool
def search_attractions(city: str) -> str:
    """Search for tourist attractions in a city."""
    if city not in ATTRACTIONS_DB:
        return "No attractions found for this city"
    
    attractions = ATTRACTIONS_DB[city]
    return f"Attractions in {city}:\nOutdoor: {', '.join(attractions['outdoor'])}\nIndoor: {', '.join(attractions['indoor'])}"

# Create tools list from decorated functions
tools = [get_weather, search_attractions]

# Create LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Create agent prompt template
prompt = PromptTemplate.from_template("""
You are a smart travel planner that considers weather conditions.
When planning a day trip:
1. ALWAYS check the weather first using get_weather
2. Based on weather:
   - If sunny/warm: Recommend outdoor activities
   - If rainy/cold: Suggest indoor activities
3. Use search_attractions to find specific places

Tools available:
{tools}

Task: {input}

{agent_scratchpad}
""")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def plan_day(city: str) -> None:
    """Plan a day trip based on weather conditions"""
    try:
        response = agent_executor.invoke({"input": f"Plan a day in {city}"})
        print(f"\nDay Plan for {city}:")
        print(response["output"])
    except Exception as e:
        print(f"Error planning trip: {str(e)}")

if __name__ == "__main__":
    # Test the agent
    plan_day("Delhi")
    print("\n" + "="*50 + "\n")
    plan_day("Paris")
