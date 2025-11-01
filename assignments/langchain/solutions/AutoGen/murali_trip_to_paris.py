import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# # Load API Key
# load_dotenv()
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in .env file")

# # ---- Common LLM Config ----
# config = {
#     "model": "gpt-4o",
#     "temperature": 0,
#     "api_key": os.getenv("OPENAI_API_KEY"),
# }

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

# Code execution config to disable Docker
code_execution_config = {
    "use_docker": False,
    "last_n_messages": 1,
    "work_dir": "workspace"
}

# ---- Define Agents ----
discoverAgent = AssistantAgent(
    name="Discoverer",
    system_message="You are a discover assistant. Research the given topic in detail and provide findings.",
    llm_config=config,
)

designAgent = AssistantAgent(
    name="Designer",
    system_message="You are an expert at planning Itinerary, Budget, Safety content preparation. Create detailed outlines from research notes for the paris trip.",
    llm_config=config,
)

ImmerseAgent = AssistantAgent(
    name="Immerser",
    system_message="You are an expert in Practice and Presence. Create content and planning for the Paris trip, focusing on immersive experiences and practical tips.",
    llm_config=config,
)

createAgent = AssistantAgent(
    name="Creator",
    system_message="You are an expert in Storytelling and Media. Prepare the document in a way that is engaging, includes captivating narratives, and suggests the inclusion of photos and media to make it visually appealing.",
    llm_config=config,
)

preserveAgent = AssistantAgent(
    name="Preserver",
    system_message="You are an expert in Sharing and Archiving. Create content to preserve memories of the Paris trip, including sharing options and archival strategies.",
    llm_config=config,
)

# User acts as the initiator
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",          #ALWAYS, NEVER, WHEN_NEEDED
    system_message="The user wants an article. cooperate to Create a document about My trip to Paris",
    code_execution_config=code_execution_config  # Add this line
)

# ---- Group Chat Setup ----
groupchat = GroupChat(
    agents=[user_proxy, discoverAgent, designAgent, ImmerseAgent, createAgent, preserveAgent],
    messages=[],
    max_round=3,   # safety stop after 10 exchanges
)

manager = GroupChatManager(groupchat=groupchat, llm_config=config)

# ---- Run Autonomous Conversation with Human in the Loop ----
while True:
    result = user_proxy.initiate_chat(
        manager,
        message=f"Create a document about My trip to Paris,"
    )

    print("\n\n=== Final Article ===\n")
    print(result)

    # Request user approval or feedback
    user_feedback = input("\nDo you approve the document? (yes/no/feedback): ").strip().lower()

    if user_feedback == "yes":
        print("\nDocument approved!")
        break
    elif user_feedback == "no":
        print("\nDocument rejected. Restarting the process...")
    else:
        print("\nFeedback received. Refining the document...")
        user_proxy.initiate_chat(
            manager,
            message=f"Refine the document based on the following feedback: {user_feedback}"
        )
