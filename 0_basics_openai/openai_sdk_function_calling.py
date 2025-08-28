from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What’s the weather in Hyderabad in celsius?"}
    ],
    tools=[{"type": "function", "function": f} for f in functions]
)

print("Function calling (SDK):")
print(response.choices[0].message)