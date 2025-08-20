import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ---- Common LLM Config ----
config = {
    "model": "gpt-4o",
    "temperature": 0,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

# ---- Define Agents ----
research_agent = AssistantAgent(
    name="Researcher",
    system_message="You are a research assistant. Research the given topic in detail and provide findings.",
    llm_config=config,
)

outline_agent = AssistantAgent(
    name="Outliner",
    system_message="You are an expert at structuring content. Create detailed outlines from research notes.",
    llm_config=config,
)

writer_agent = AssistantAgent(
    name="Writer",
    system_message="You are a professional content writer. Write a full article from a given outline.",
    llm_config=config,
)

# User acts as the initiator
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    system_message="The user wants an article. Cooperate to research, outline, and write the article."
)

# ---- Group Chat Setup ----
groupchat = GroupChat(
    agents=[user_proxy, research_agent, outline_agent, writer_agent],
    messages=[],
    max_round=10,   # safety stop after 10 exchanges
)

manager = GroupChatManager(groupchat=groupchat, llm_config=config)

# ---- Run Autonomous Conversation ----
topic = "Impact of AI in Education"
result = user_proxy.initiate_chat(
    manager,
    message=f"Generate a detailed article on: {topic}. "
            f"Follow these steps: research -> outline -> article. "
            f"Collaborate among yourselves to produce the final article."
)

print("\n\n=== Final Article ===\n")
print(result)