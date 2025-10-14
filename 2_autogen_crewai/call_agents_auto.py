import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# # Load API Key
# load_dotenv()
# if not os.getenv("GROQ_API_KEY"):
#     raise ValueError("GROQ_API_KEY not found in .env file")

# # ---- Common LLM Config ----
# llm_config = {
#     "model": "llama-3.1-8b-instant", 
#     "api_key": os.getenv("GROQ_API_KEY"),
#     "base_url": "https://api.groq.com/openai/v1"
# }


# load api key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


# llm config
llm_config = {
    "model": "gpt-4o",
    "temperature": 0,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

# ---- Define Diagnostic Agents ----

wifi_agent = AssistantAgent(
    name="WiFiAgent",
    system_message="You are a network specialist. Check if Wi-Fi activity, background updates, or downloads are affecting power usage.",
    llm_config=llm_config,
)

battery_agent = AssistantAgent(
    name="BatteryAgent",
    system_message="You are a battery diagnostics expert. Analyze battery health, charging status, and power draw.",
    llm_config=llm_config,
)

screen_agent = AssistantAgent(
    name="ScreenExpertAgent",
    system_message="You are a display technician. Evaluate screen brightness, resolution, and display power consumption.",
    llm_config=llm_config,
)

# ---- User Proxy Agent ----
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    system_message="The user wants to know why their laptop hasn't charged in the past hour. Collaborate to diagnose the issue.",
)

# ----  Group Chat ----
groupchat = GroupChat(
    agents=[user_proxy, wifi_agent, battery_agent, screen_agent],
    messages=[],
    max_round=2,
)

groupchat.agent_selection_method = "auto"


manager = GroupChatManager(groupchat=groupchat,
                           llm_config=llm_config)

# ---- Start the Diagnostic Conversation ----
user_proxy.initiate_chat(
    manager,
    message="My laptop has been plugged in for an hour but hasn't charged. Please investigate possible causes."
)
