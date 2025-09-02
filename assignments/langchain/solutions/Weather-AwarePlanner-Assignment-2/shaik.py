import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,AgentType

#load env

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found")

# define tools
@tool
def get_weather(city:str)->str:
    """Get current weather for a given city"""
    mock_weather={
        "Delhi":"Sunny 28° C",
        "Paris":"Rainy 18° C",
        "London":"Cloudy 20° C",
        
    }
    return mock_weather.get(city,"Sunny 25° C")

@tool
def search_attractions(city:str)->str:
    """Search for top tourist attractions in a given city"""
    mock_attractions={
        "Delhi":["India Gate","Qutub Minar","Lotus Temple"],
        "Paris":["Eiffel Tower","Louvrs Museum","Notre Dame"],
        "London":["London Eye","British Museum","Tower of London"]
    }

    return ",".join(mock_attractions.get(city,["Local Park","Museum"]))

#collect tools
tools=[get_weather,search_attractions]

#initialize LLM
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)

#Initialize Agent
agent=initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

#run query
query="plan a day in Paris"
response=agent.run(query)

print("\n---Final Plan ---")
print(response)