import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from llama_index.llms.openai import OpenAI

# Load API Key
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")

# ---- Common LLM Config ----
config = {
    "model": "llama-3.3-70b-versatile", 
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1"
}

# mixtral-8x7b-32768, llama3-70b-8192, or gemma-7b-it) -- decomissioned
# General-purpose: llama-3.3-70b-versatile, gpt-oss-120b
# Fast & lightweight: llama-3.1-8b-instant, gpt-oss-20b

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
    max_round=2,   # safety stop after 10 exchanges
)

manager = GroupChatManager(groupchat=groupchat, llm_config=config)

# ---- Run Autonomous Conversation ----
topic = "Impact of AI in Education"
result = user_proxy.initiate_chat(
    manager,
    message=f"Generate a detailed article on: {topic}. \n"
            f"Follow these steps: research -> outline -> article. \n"
            f"Collaborate among yourselves to produce the final article.\n"
)

print("\n\n=== Final Article ===\n")
print(result)