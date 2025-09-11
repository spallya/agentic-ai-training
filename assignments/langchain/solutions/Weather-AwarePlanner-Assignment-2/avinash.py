import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,AgentType

#loadEnv
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found")

#Tools
@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city (mocked)"""
    mock_weather = {
        "Delhi":"Sunny 28° C",
        "Paris":"Rainy 20° C",
    }
    return mock_weather.get(city, "Weather data not available.")

@tool
def search_attractions(city: str) -> str:
    """Search for some tourist attractions in a given city"""
    mock_attractions = {
        "Delhi": ["India Gate", "Red Fort", "Akshardham"],
        "Paris": ["Eiffel Tower", "Arc De Triomphe", "Louvre Museum"]
    }
    return ", ".join(mock_attractions[city])

@tool
def suggest_activity(weather: str) -> str:
    """Suggest indoor/outdoor activity from attractions based on the weather condition."""
    if "Sunny" in weather:
        return "suggest outdoor activities from attractions tool"
    if "Rainy" in weather:
        return "suggest indoor activities"
    return "suggest indoor/outdoor activity"


#SetupLLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#SetupAgent
tools = [get_weather, search_attractions, suggest_activity]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("\n--- Weather Assistant ---\n")
    result = agent.invoke("plan a day in Bangalore")
    print("\nFinal result:", result)