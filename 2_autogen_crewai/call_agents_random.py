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

# ---- Define Agents ----

joke_agent = AssistantAgent(
    name="JokeAgent",
    system_message="You are a witty comedian. Add humor and jokes related to AI and technology.",
    llm_config=llm_config,
)

poem_agent = AssistantAgent(
    name="PoemAgent",
    system_message="You are a poetic soul. Contribute poetic reflections on the AI world.",
    llm_config=llm_config,
)

info_agent = AssistantAgent(
    name="InfoAgent",
    system_message="You are a knowledgeable analyst. Provide factual and structured information about AI.",
    llm_config=llm_config,
)

# ---- User Proxy Agent ----
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    system_message="The user wants an essay on the AI world. Collaborate creatively and informatively.",
)

# ---- Randomized Group Chat ----
agents = [user_proxy, joke_agent, poem_agent, info_agent]

groupchat = GroupChat(
    agents=agents,
    messages=[],
    max_round=4,
)

groupchat.agent_selection_method = "random"

manager = GroupChatManager(groupchat=groupchat,
                           llm_config=llm_config,
                            human_input_mode="NEVER")

# ---- Initiate Chat ----
user_proxy.initiate_chat(
    manager,
    message="Write an essay story AI todays trend. Include jokes, poems, and informative content."
)
