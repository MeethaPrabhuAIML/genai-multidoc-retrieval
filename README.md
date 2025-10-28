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

# urls = [
#     "https://openreview.net/pdf?id=5atraF1tbg",
#     "https://openreview.net/pdf?id=YaEozn3y0G",
#     "https://openreview.net/pdf?id=P6NcRPb13w",
# ]

papers = [
    "privacy.pdf",
    "ml_topo.pdf",
    "ml4.pdf",
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
    "Tell me about the Topology used in Machine Learning, "
    "and then tell me about privacy"
)

response = agent.query("Give me a summary of both Adjusting Machine Learning and Topology in Machine Learning")
print(str(response))

# urls = [
#     "https://openreview.net/pdf?id=VtmBAGCN7o",
#     "https://openreview.net/pdf?id=6PmJoRfdaK",
#     "https://openreview.net/pdf?id=LzPWWPAdY4",
#     "https://openreview.net/pdf?id=VTF8yNQM66",
#     "https://openreview.net/pdf?id=hSyW5go0v8",
#     "https://openreview.net/pdf?id=9WD9KwssyT",
#     "https://openreview.net/pdf?id=yV6fD7LYkF",
#     "https://openreview.net/pdf?id=hnrB5YHoYu",
#     "https://openreview.net/pdf?id=WbWtOYIzIK",
#     "https://openreview.net/pdf?id=c5pwL0Soay",
#     "https://openreview.net/pdf?id=TpD2aG1h0D"
# ]

papers = [
    "ml2.pdf",
    "ml3.pdf",
    "ml4.pdf",
    "ML_blockchain.pdf",
    "ML_packages.pdf",
    "ML_pipelines.pdf",
    "ml_topo.pdf",
    "privacy.pdf",
    "MachineLearning.pdf",
    "memorization.pdf",
    "ai.pdf"
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve(
    "Tell me about the formal definition of memorisation used in Machine Learning and Regularization"
)

tools[2].metadata

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the formal definition of memorisation used" 
    "in Machine Learning and Regularization"
)
print(str(response))
```
### OUTPUT:
<img width="1077" height="812" alt="image" src="https://github.com/user-attachments/assets/3450f54d-1ca8-4168-a7e5-79b417f7b2a7" />

<img width="957" height="602" alt="image" src="https://github.com/user-attachments/assets/83821936-ce01-4aff-8787-326b9220a37e" />

<img width="981" height="755" alt="image" src="https://github.com/user-attachments/assets/fabc4fff-ebea-4da3-b868-1feefd791613" />

### RESULT:
Thus, the experiment successfully demonstrated how to build a scalable agent that dynamically retrieves specific document tools from a searchable index to answer user queries successfully.

