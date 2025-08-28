from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")
# Make sure you set your API key in environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",   # You can change the model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about the ocean."}
    ],
    max_tokens=100
)

print("OpenAI library response:")
print(response.choices[0].message.content)