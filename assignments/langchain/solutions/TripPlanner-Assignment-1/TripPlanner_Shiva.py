import os
import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence

# === ENV SETUP ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# === GROQ CONFIG ===
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# === PURE FUNCTION: Groq Chat Wrapper ===
def groq_chat(prompt: str) -> str:
    """Sends a prompt to Groq's LLM and returns the response."""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error contacting Groq API: {e}"

# === STEP 1: Research Attractions ===
def research_attractions(city: str) -> str:
    return f"List 3 top tourist attractions in {city} with a brief description of each."

# === STEP 2: Summarize as Bullet Points ===
def summarize_attractions(attractions_text: str) -> str:
    return f"Summarize the following into bullet points:\n\n{attractions_text}"

# === STEP 3: Generate 1-Day Itinerary ===
def generate_itinerary(summary_text: str) -> str:
    return f"Using the following attractions, generate a short 1-day itinerary:\n\n{summary_text}"

# === BUILD PIPELINE ===
pipeline = RunnableSequence(
    first=RunnableLambda(lambda city: city.strip()),
    middle=[
        RunnableLambda(research_attractions),
        RunnableLambda(groq_chat),
        RunnableLambda(summarize_attractions),
        RunnableLambda(groq_chat),
        RunnableLambda(generate_itinerary),
        RunnableLambda(groq_chat)
    ],
    last=RunnableLambda(lambda itinerary: itinerary.strip())
)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    city = input("Enter the destination city: ")
    result = pipeline.invoke(city)
    print("\n🗺️ Your 1-Day Travel Itinerary:\n")
    print(result)