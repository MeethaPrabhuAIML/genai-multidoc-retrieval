## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

To create an intelligent agent that can answer questions about a large collection of documents. The key challenge is to avoid overwhelming the agent by giving it all document-specific tools at once, and instead enable it to dynamically retrieve only the necessary tools (like "query" or "summary" for a specific paper) based on the user's question.

### DESIGN STEPS:

#### STEP 1:
Create vector search and summary tools for a small set of 3 papers. Then, build a basic agent by giving it this fixed list of 6 tools.

#### STEP 2:
Generate the same types of tools (vector and summary) for a much larger collection of 11 different papers, resulting in 22 tools.

#### STEP 3:
Create a special searchable index (ObjectIndex) from all 22 tools. This index allows the agent to search for and find the right tool for a specific query.

#### STEP 4:
Build a new, advanced agent that uses a "tool retriever" instead of a fixed list. This retriever searches the ObjectIndex to find the best tools for the user's question.

#### STEP 5:
Test the advanced agent by asking questions about specific topics. The agent now automatically finds and uses only the relevant tools from the 11 papers to answer.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/pdf?id=XmProj9cPs",
    "https://openreview.net/pdf?id=NGKQoaqLpo",
    "https://openreview.net/pdf?id=odjMSBSWRt",
]

papers = [
    "spider.pdf",
    "llmknowledgedilute.pdf",
    "darkbench.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

len(initial_tools)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Why was Spider 2.0 created, and what problem does it solve in enterprise text-to-SQL workflows?"
)

response = agent.query(
    "How does new surprising text affect LLMs’ existing knowledge, and how can this impact be controlled?"
)

response = agent.query(
    "How prevalent are dark patterns in LLMs and chatbots, and what benchmarks or methods exist to detect and mitigate them?"
)



```
### OUTPUT:
<img width="1077" height="812" alt="image" src="https://github.com/user-attachments/assets/3450f54d-1ca8-4168-a7e5-79b417f7b2a7" />

<img width="957" height="602" alt="image" src="https://github.com/user-attachments/assets/83821936-ce01-4aff-8787-326b9220a37e" />

<img width="981" height="755" alt="image" src="https://github.com/user-attachments/assets/fabc4fff-ebea-4da3-b868-1feefd791613" />

### RESULT:
Thus, the experiment successfully demonstrated how to build a scalable agent that dynamically retrieves specific document tools from a searchable index to answer user queries successfully.

