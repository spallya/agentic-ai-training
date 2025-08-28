from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define available functions (tools)
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

# Call Groq API with tool definitions
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Use Groq model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What’s the weather in Hyderabad in celsius?"}
    ],
    tools=[{"type": "function", "function": f} for f in functions]
)

# Print function call result
print("Function calling (Groq SDK):")
print(response.choices[0].message)