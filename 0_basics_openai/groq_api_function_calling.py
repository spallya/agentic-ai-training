import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"

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
    "model": "llama-3.3-70b-versatile",  # Updated to supported model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What’s the weather in Hyderabad in celsius?"}
    ],
    "tools": [{"type": "function", "function": f} for f in functions]
}

response = requests.post(url, headers=headers, json=data)
response_json = response.json()

print("Function calling (Groq API):")
print(json.dumps(response_json, indent=2))

# Only attempt to access 'choices' if response succeeded
if "choices" in response_json:
    tool_calls = response_json["choices"][0]["message"].get("tool_calls")
    if tool_calls:
        print("\nSuggested function call:")
        print(json.dumps(tool_calls, indent=2))
else:
    print("\nNo valid response received. Please check the error above.")