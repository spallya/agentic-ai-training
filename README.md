# Agentic AI Training
*Hands-on playground to build, test, and evolve agent-based AI systems.*  

**Agentic AI Training** is a hands-on Python sandbox for experimenting with **agent-based AI workflows**.

From building autonomous content generators to orchestrating workflows with **LangChain**, **LangGraph**, and **MCP**, this repo helps you **prototype, learn, and evolve real-world AI agent systems**.

### 🔑 Highlights
- 🛠 **Real-world scaffolding** – end-to-end examples like article writer agents, MCP servers, and graph-based orchestration.  
- 🔗 **Framework integration** – play with LangChain, Autogen, Crewai, LangGraph, and the emerging MCP protocol.  
- ⚡ **Modular & practical** – simple structure, easy to adapt and extend.  
- 🎯 **Learn by doing** – tweak the code, run the agents, and explore agentic AI in action.  

---

## Table of Contents

- [Overview](#overview)  
- [Key Components](#key-components)  
- [Features](#features)  
- [Getting Started](#getting-started)  
- [Developer Setup](#developer-setup)  
- [Usage Examples](#usage-examples)  
- [Visuals / Demo](#visuals--demo)  
- [Learn More](#learn-more)
- [Assignment Submission Guidelines](#assignment-submission-guidelines)  
- [Contributing](#contributing)  
- [License](#license)

---

## Overview

This repository mirrors the content of the live **Agentic AI training** sessions—offering developers, from beginners to advanced, a sandbox to build, test, and learn through intelligent agent workflows.

---

## Key Components

- **Article Writer Modules**  
  - `article_writer_autogen.py`: Auto-generation pipelines.  
  - `article_writer_langchain_tools.py`: Integrating Langchain for content creation.
- **Lang chain core concepts with programs**
  - For LangChain Theoretical concepts, go through PresentationLangChain ppt in docs folder and explore practical programs.
  - `react_design_pattern_agent_tool_selector` :  
    Demonstrates how the **ReAct Pattern/Agent** dynamically selects tools based on a user query and returns a complete answer covers concepts oftools, ReAct Pattern and llm wrapper.
  - `react_design_pattern_agent_with_buffer_memory` :  
    Shows how conversation memory is used (e.g., in chatbots, short-term memory).  
    Example usage:  
    - Question 1: *Select flights in Delhi*  
    - Question 2: *Does it have good hotels?*  
    (Notice that you don’t need to repeat “Delhi”; memory handles the context.)  
  - `rag_pinecone_pdf_demo` :  
  This RAG example uses **Pinecone** as the vector database (retriever) and **GPT-4** as the generator.  
  Set up all required keys in the `.env` file:  
  `(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, FORCE_RECREATE_INDEX)`.  
  Follow pinecone connection setup document in docs folder.
  Run the program and ask questions about the document you ingested.
  - `react_with_knowledgebase` :  
    Illustrates how LLM responses can be stored in a **local vector DB/knowledge base** to avoid calling the LLM every time.  
    Example:  
    - Question 1: *Who is Elon Musk?* → Answer from LLM (and stored).  
    - Question 2: *Tell me about Elon Musk again* → Answer retrieved from the knowledge base (not the LLM).  

  - `vector_search_with_images` :  
    Example of how a vector database stores and retrieves **image embeddings**.  
    Place some images in the `images` folder, then query: *red running shoes*.  
    The program will return the matching image(s).  

  - `rag_confluence_example` :  
    In this RAG example, the retriever is **Confluence**, and the generator is an LLM.  
    Configure your Confluence environment variables:  
    `(BASE_URL, EMAIL, API_TOKEN)`.
    Follow confluence setup document in docs folder
    Run the program and ask questions directly from your Confluence pages—the agent/LLM will answer using that content.

- **MCP Servers**  
  - `bmi_mcp_server.py`, `math_mcp_server.py`: Demonstrations of setting up Multi-Context Processing servers for structured data handling.

- **Langgraph Integration**  
  - `langgraph_with_multiple_mcp.py`, `mcp_with_langgraph.py`: Examples of combining Langchain with MCP servers for more advanced agent orchestration.

- **Client Interaction**  
  - `mcp_client.py`: Sample client for real-time interaction with MCP servers.

---

## Features

- Build and experiment with intelligent AI agents in Python.
- Leverage Langchain together with robust MCP server architectures.
- Hands-on scripts for content generation and orchestrated workflows.
- Modular and extensible design tailored for experimentation.

---

## Getting Started

To run locally:

```bash
git clone https://github.com/spallya/agentic-ai-training.git
cd agentic-ai-training
pip install -r requirements.txt
```

Explore the script files to understand how to build and extend agentic AI systems.

---

## Developer Setup

### IDEs
- Recommended: **VS Code** or **PyCharm**
- Both IDEs provide excellent support for Python virtual environments and `.env` configuration.

### Create Virtual Environment
```bash
# Create venv
python -m venv .venv

# Activate venv (Linux / Mac)
source .venv/bin/activate

# Activate venv (Windows PowerShell)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys Setup

1. **OpenAI API Key**
   - Sign up / log in at [OpenAI](https://platform.openai.com/).  
   - Generate an API key from the [API Keys dashboard](https://platform.openai.com/account/api-keys).

2. **Groq API Key**
   - Sign up / log in at [Groq](https://console.groq.com/).  
   - Generate an API key from the dashboard.

### Configure `.env` File

Create a `.env` file in the project root with the following content:

```ini
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

> The code automatically loads these values when running scripts.

---

## Usage Examples

```bash
# Generate an article using the auto-gen agent
python article_writer_autogen.py

# Launch an MCP server (e.g., math context processing)
python math_mcp_server.py

# Connect to an MCP server with the client
python mcp_client.py
```

---

## 📸 Visuals / Demo

<table>
  <tr>
    <td align="center" valign="top">
      <b>Article Writer Graph</b><br>
      <img src="graphs/article_writer_langgraph.png" alt="Article writer graph" width="250" height="450">
    </td>
    <td align="center" valign="top">
      <b>Graph with Tools and Conditional Edge</b><br>
      <img src="graphs/langgraph_with_multiple_mcp.png" alt="Graph with MCP" width="250" height="300">
    </td>
  </tr>
</table>
---

## Learn More

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/introduction)  
- [Langchain Documentation](https://langchain.com)  
- [Why Langchain?](https://python.langchain.com/docs/concepts/why_langchain/)  
- [MCP Framework Overview](https://modelcontextprotocol.io)  

---
# Assignment Submission Guidelines

To submit your assignments, please follow the steps below:

## Folder Structure

Your assignment should be placed under the `assignments/` folder in the following structure:

```
assignments/
└── <assignment-topic>/
    └── solutions
        └── <your-name>.py
```

### Example
```
assignments/
└── langchain/
    └── solutions
        └── spallya-omar.py
```

---

## Submission Process

1. **Fork the Repository**
   - Click on the **Fork** button on the top right of this repository.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/<your-username>/agentic-ai-training.git
   cd agentic-ai-training
   ```

3. **Create a Branch**
   ```bash
   git checkout -b langchain-submission
   ```

4. **Add Your Assignment**
   - Place your solution file under the correct folder:  
     `assignments/langchain/solutions/<your-name>.py`

5. **Commit and Push**
   ```bash
   git add assignments/langchain/solutions/<your-name>.py
   git commit -m "Adding langchain assignment solution by <your-name>"
   git push origin assignment1-submission
   ```

6. **Create a Pull Request**
   - Go to your forked repository on GitHub.
   - Click on **Compare & Pull Request**.
   - Provide a meaningful title and description, then submit the PR.

### Notes
- Make sure your filename matches the required format (`<your-name>.py`).
- Ensure your code runs without errors before submission.
- Only one submission file per participant.


---
## Contributing

Contributions are highly valued! If you'd like to suggest enhancements, fix bugs, or propose new features, please fork the repo and submit a pull request.

---

## License

Licensed under the **MIT License**—see the [LICENSE](LICENSE) file for full details.

