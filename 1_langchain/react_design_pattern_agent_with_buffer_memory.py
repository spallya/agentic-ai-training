from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, tool
from langchain.memory import ConversationBufferMemory

load_dotenv()

# --- Define tools ---
@tool
def flight_info(city: str) -> str:
    """Return mock flight info to a given city."""
    city = city.lower()
    flights = {
        "delhi": "✈️ 3 daily flights, approx 2 hrs, avg fare ₹5000",
        "mumbai": "✈️ 5 daily flights, approx 1.5 hrs, avg fare ₹4000",
        "bangalore": "✈️ 4 daily flights, approx 2.5 hrs, avg fare ₹5500",
    }
    return flights.get(city, "No flight info available.")

@tool
def hotel_info(city: str) -> str:
    """Return mock hotel info for a city."""
    hotels = {
        "delhi": "5⭐ Taj Palace, 4⭐ ITC Maurya",
        "mumbai": "5⭐ The Oberoi, 4⭐ Trident Nariman Point",
        "bangalore": "5⭐ Leela Palace, 4⭐ ITC Gardenia",
    }
    return hotels.get(city.lower(), "No hotel info available.")

# --- Setup LLM + Memory ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [flight_info, hotel_info]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    print("\n--- Travel Assistant with Memory ---")
    print("Type your question (or 'q' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["q", "quit", "exit"]:
            print("Goodbye! 👋")
            break
        out = agent.invoke({"input": q})
        print("Agent:", out["output"], "\n")
