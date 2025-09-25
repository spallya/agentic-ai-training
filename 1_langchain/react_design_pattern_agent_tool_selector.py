import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, tool

# --- Load env (needs OPENAI_API_KEY) ---
load_dotenv()

# --- Define tools for ONE example (Weather helper) ---
@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city (mocked)."""
    city = city.lower().strip()
    fake_data = {
        "hyderabad": "☀️ Sunny, 32°C",
        "bangalore": "🌧️ Rainy, 24°C",
        "delhi": "🌤️ Partly cloudy, 30°C",
    }
    return fake_data.get(city, "Weather data not available.")

@tool
def suggest_outfit(weather: str) -> str:
    """Suggest an outfit based on the weather string."""
    if "Sunny" in weather:
        return "Wear light cotton clothes and sunglasses 😎"
    if "Rainy" in weather:
        return "Carry an umbrella ☔ and wear waterproof shoes."
    if "cloudy" in weather.lower():
        return "A light jacket would be good."
    return "Dress comfortably."

# --- Setup LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Setup Agent ---
tools = [get_weather, suggest_outfit]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct, scratchpad handled internally
    verbose=True
)

if __name__ == "__main__":
    print("\n--- Weather Assistant ---\n")
    result = agent.run("I am in Hyderabad, can you suggest what I should wear today?")
    print("\nFinal Answer:", result)
