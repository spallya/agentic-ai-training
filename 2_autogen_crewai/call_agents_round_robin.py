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


# ---- Define Cake Baking Agents ----

ingredient_agent = AssistantAgent(
    name="IngredientMaster",
    system_message="You are an expert in selecting ingredients. List all ingredients needed for baking a classic vanilla cake.",
    llm_config=llm_config,
)

prep_agent = AssistantAgent(
    name="PrepChef",
    system_message="You are a prep chef. Describe how to prepare the ingredients for baking.",
    llm_config=llm_config,
)

bake_agent = AssistantAgent(
    name="OvenWizard",
    system_message="You are a baking expert. Explain how to bake the cake step-by-step.",
    llm_config=llm_config,
)

decorate_agent = AssistantAgent(
    name="CakeDecorator",
    system_message="You are a cake decorator. Suggest creative ways to decorate the cake once it's baked.",
    llm_config=llm_config,
)

# ---- User Proxy Agent ----
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    system_message="The user wants to bake a cake. Collaborate step-by-step to complete the task.",
)

# ---- Round-Robin Group Chat ----
groupchat = GroupChat(
    agents=[user_proxy, ingredient_agent, prep_agent, bake_agent, decorate_agent],
    messages=[],
    max_round=5,
    send_introductions=True
)
groupchat.agent_selection_method = "round_robin"

manager = GroupChatManager(groupchat=groupchat,
llm_config=llm_config
)

# ---- Initiate Chat ----
user_proxy.initiate_chat(
    manager,
    message="Let's bake a vanilla cake together. Start with ingredients and go step-by-step until it's decorated."
)
