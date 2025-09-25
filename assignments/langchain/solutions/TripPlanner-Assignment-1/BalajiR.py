from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize the LLM with OpenAI's model
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"  # Using GPT-3.5 Turbo model
)

# Create prompt templates
research_prompt = ChatPromptTemplate.from_template("""
Please research and list exactly 3 top tourist attractions in {city}.
Provide detailed information about each attraction.
""")

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following attractions into concise bullet points:
{research_result}
""")

itinerary_prompt = ChatPromptTemplate.from_template("""
Based on these attractions:
{summary_result}

Create a practical 1-day itinerary that includes:
- Recommended timing for each visit
- Logical order of visits
- Transportation suggestions between locations
- Meal recommendations
Keep it realistic and include brief breaks.
""")

# Create the chain
chain = (
    {
        "research_result": research_prompt | llm,
        "city": lambda x: x
    }
    | {
        "summary_result": lambda x: summary_prompt | llm | (lambda y: y.content),
        "city": lambda x: x["city"]
    }
    | itinerary_prompt
    | llm
)

def plan_trip(city: str) -> str:
    """Generate a trip plan for the specified city."""
    try:
        result = chain.invoke(city)
        return result.content
    except Exception as e:
        return f"Error generating trip plan: {str(e)}"

if __name__ == "__main__":
    # Example usage
    city = input("Enter a destination city: ")
    plan = plan_trip(city)
    print("\n=== Your Trip Plan ===\n")
    print(plan)