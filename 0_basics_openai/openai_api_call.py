import requests
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Make sure you set your API key in environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Example: Calling Chat Completions API directly
url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "gpt-4o-mini",   # You can change the model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about the ocean."}
    ],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)

print("Direct API response:")
print(response.json()["choices"][0]["message"]["content"])