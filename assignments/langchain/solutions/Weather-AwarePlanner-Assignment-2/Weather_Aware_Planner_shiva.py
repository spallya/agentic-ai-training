import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

# === Load Environment Variables ===
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found")

# === Tool 1: Get Weather ===
@tool
def get_weather(city: str) -> str:
    """Returns mock weather string for a given city."""
    mock_weather = {
        "Delhi": "Sunny 28°C",
        "Paris": "Rainy 20°C",
        "London": "Cloudy 22°C",
        "Tokyo": "Sunny 30°C",
        "New York": "Rainy 18°C",
        "Hyderabad": "Partly Cloudy 27°C",
        "Dubai": "Hot 38°C",
        "Berlin": "Windy 16°C"
    }
    return mock_weather[city]

# === Tool 2: Search Attractions ===
@tool
def search_attractions(city: str) -> str:
    """Returns mock tourist attractions for a given city."""
    mock_attractions = {
        "Delhi": ["India Gate", "Lodhi Gardens", "Red Fort"],
        "Paris": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
        "London": ["London Eye", "British Museum", "Tower of London"],
        "Tokyo": ["Senso-ji Temple", "Shibuya Crossing", "Tokyo Skytree"],
        "New York": ["Central Park", "Metropolitan Museum", "Times Square"],
        "Hyderabad": ["Charminar", "Golconda Fort", "Hussain Sagar Lake"],
        "Dubai": ["Burj Khalifa", "Dubai Mall", "Palm Jumeirah"],
        "Berlin": ["Brandenburg Gate", "Museum Island", "Berlin Wall Memorial"]
    }
    return ", ".join(mock_attractions[city])

# === Tool 3: Plan Strategy Based on Weather ===
@tool
def plan_strategy(weather: str) -> str:
    """Suggests planning strategy based on weather string like 'Sunny 28°C'."""
    condition = weather.lower()

    if "sunny" in condition or "hot" in condition:
        return "Since the weather is clear and warm, outdoor sightseeing is ideal."
    elif "cloudy" in condition or "windy" in condition:
        return "With overcast skies, a mix of indoor and outdoor activities works well."
    elif "rainy" in condition:
        return "Due to rain, indoor attractions are recommended to stay dry and comfortable."
    else:
        return "Weather is moderate, so a balanced itinerary is suggested."

# === Collect Tools ===
tools = [get_weather, search_attractions, plan_strategy]

# === Initialize Groq LLM ===
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# === Initialize Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === Supported Cities ===
available_cities = {
    "Delhi", "Paris", "London", "Tokyo", "New York",
    "Hyderabad", "Dubai", "Berlin"
}

# === City Input Loop ===
while True:
    user_city = input("🌍 Enter a city you'd like to explore: ").strip().title()

    if user_city in available_cities:
        query = f"Plan a day in {user_city}"
        response = agent.invoke(query)

        # Get weather and strategy
        weather_info = get_weather.run(user_city)
        strategy = plan_strategy.run(weather_info)

        # Display formatted output
        print(f"\n--- Explore {user_city} with this plan ---")
        print(f"🌦️ Weather: {weather_info}")
        print(f"🧭 Strategy: {strategy}")
        print(f"\n📍 Itinerary: {response['output']}")
        break

    else:
        print(f"\n😕 Sorry, I don’t have data for “{user_city}” yet.")
        print("I’m currently set up to plan trips for these cities:")
        print("🗺️ " + ", ".join(sorted(available_cities)))
        print("Please enter one of the cities listed above to continue.\n")