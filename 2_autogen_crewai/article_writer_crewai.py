import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ---- Common LLM Config ----
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---- Define Agents ----
research_agent = Agent(
    role="Research Assistant",
    goal="Research the given topic in detail and provide findings.",
    backstory="You are excellent at researching topics and summarizing detailed insights.",
    llm=llm
)

outline_agent = Agent(
    role="Outliner",
    goal="Create detailed structured outlines from research notes.",
    backstory="You are skilled at organizing research into structured outlines.",
    llm=llm
)

writer_agent = Agent(
    role="Content Writer",
    goal="Write a full professional article from a given outline.",
    backstory="You are a professional writer with experience in turning outlines into polished articles.",
    llm=llm
)

# ---- Define Tasks ----
topic = "Impact of AI in Education"

research_task = Task(
    description=f"Research in detail about: {topic}. Provide comprehensive findings.",
    agent=research_agent,
    expected_output="Detailed research notes on the topic."
)

outline_task = Task(
    description="Take the research notes and create a structured outline for the article.",
    agent=outline_agent,
    expected_output="A clear, detailed outline with sections and bullet points."
)

writing_task = Task(
    description="Expand the outline into a professional full-length article.",
    agent=writer_agent,
    expected_output="A final article ready for publishing."
)

# ---- Crew Setup ----
crew = Crew(
    agents=[research_agent, outline_agent, writer_agent],
    tasks=[research_task, outline_task, writing_task],
    verbose=True
)

# ---- Run Autonomous Workflow ----
result = crew.kickoff()

print("\n\n=== Final Article ===\n")
print(result)