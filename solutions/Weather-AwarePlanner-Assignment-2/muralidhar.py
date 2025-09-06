import os
import json
from typing import Dict, Any, List
from openai import OpenAI

# --------------- Mock Tools ---------------

def get_weather(city: str) -> str:
    """
    Mock weather: return "Sunny 28°C" for Delhi, "Rainy 20°C" for Paris, else "Unknown".
    """
    c = city.strip().lower()
    if c == "delhi":
        return "Sunny 28°C"
    if c == "paris":
        return "Rainy 20°C"
    return "Unknown"

def search_attractions(city: str) -> Dict[str, List[str]]:
    """
    Mock attractions: returns indoor and outdoor attractions for Delhi and Paris.
    """
    c = city.strip().lower()
    if c == "delhi":
        return {
            "outdoor": ["India Gate", "Lodhi Gardens", "Qutub Minar"],
            "indoor": ["National Museum", "Rail Museum", "DLF Mall of India"],
        }
    if c == "paris":
        return {
            "outdoor": ["Champ de Mars (Eiffel Tower lawns)", "Luxembourg Gardens", "Montmartre Walk"],
            "indoor": ["Louvre Museum", "Musée d'Orsay", "Sainte-Chapelle"],
        }
    return {
        "outdoor": ["City Park", "Riverside Walk", "Historic Old Town"],
        "indoor": ["City Museum", "Art Gallery", "Science Center"],
    }

# --------------- OpenAI Tool Schemas ---------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city (mocked).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g., 'Delhi' or 'Paris'."}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Return indoor and outdoor attractions (mocked).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g., 'Delhi' or 'Paris'."}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

# Map tool names to Python callables
available_functions = {
    "get_weather": get_weather,
    "search_attractions": search_attractions,
}

# --------------- Agent Runner ---------------

SYSTEM_INSTRUCTIONS = (
    "You are a helpful weather-aware day planner. "
    "When the user asks to plan a day in a city, ALWAYS call get_weather(city) first. "
    "If the weather is sunny, recommend primarily outdoor activities from search_attractions(city). "
    "If the weather is rainy, recommend primarily indoor activities from search_attractions(city). "
    "Cite the weather in your plan and tailor the itinerary to the conditions."
)

def run_agent(user_input: str) -> str:
    client = OpenAI()  # uses OPENAI_API_KEY from the environment
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_input},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices.message

        # If the model wants to call tools, execute them and return results
        if msg.tool_calls:
            # It’s important to include the assistant message with tool_calls in the history
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })

            for tool_call in msg.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    function_args = {}

                if function_name in available_functions:
                    result = available_functions[function_name](**function_args)
                else:
                    result = f"Error: unknown function {function_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # Loop back so the model can incorporate tool results and produce final answer
            continue

        # No tool call => final assistant message
        return msg.content or "(No content returned.)"

if __name__ == "__main__":
    # Examples
    print("Delhi plan:\n", run_agent("Plan a day in Delhi"), "\n")
    print("Paris plan:\n", run_agent("Plan a day in Paris"), "\n")
