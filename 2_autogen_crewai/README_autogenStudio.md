# Prereqs: Python 3.10+ (use a venv), pip.

# How to install Autogen studio
pip install -U autogenstudio

# export your open ai key
export OPENAI_API_KEY = "your key here"

# To run autogen studio
autogenstudio ui --port 8080
# Open http://localhost:8080 (or the port you chose).


# any issues set
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# run backend with hot reload
autogenui --reload      
# runs backend + serves prebuilt frontend by default on :8081



