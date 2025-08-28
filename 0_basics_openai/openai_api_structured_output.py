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

data = {
    "model": "gpt-4.1",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract a structured user profile: My name is Omar, I am 32 years old, and I live in Hyderabad."}
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "user_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age", "city"],
                "additionalProperties": False
            }
        }
    }
}

response = requests.post(url, headers=headers, json=data).json()

print("Full raw response:")
print(json.dumps(response, indent=2))

parsed = json.loads(response["choices"][0]["message"]["content"])
print("\nParsed structured output:")
print(parsed)