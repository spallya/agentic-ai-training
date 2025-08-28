import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

api_key = os.getenv("OPENAI_API_KEY")

url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Name of the city"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What’s the weather in Hyderabad in celsius?"}
    ],
    "tools": [{"type": "function", "function": f} for f in functions]
}

response = requests.post(url, headers=headers, json=data).json()

print("Function calling (direct API):")
print(json.dumps(response, indent=2))

# Check if tool call was suggested
tool_calls = response["choices"][0]["message"].get("tool_calls")
if tool_calls:
    print("Suggested function call:", tool_calls[0])