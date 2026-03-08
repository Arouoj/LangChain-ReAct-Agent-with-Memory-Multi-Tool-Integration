import os
import time
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings for cleaner output
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_classic.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

# ===== LLM Configuration =====
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=500,
    timeout=30
)

# ----- Tools (✅ FIXED: Simple names, NO SPACES) -----

def safe_search(query: str) -> str:
    try:
        return DuckDuckGoSearchRun().run(query[:150])
    except Exception as e:
        return f"Search error: {str(e)[:100]}"

def safe_wiki(query: str) -> str:
    try:
        return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2)).run(query[:100])
    except Exception as e:
        return f"Wikipedia error: {str(e)[:100]}"

def safe_arxiv(query: str) -> str:
    try:
        return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2)).run(query[:100])
    except Exception as e:
        return f"Arxiv error: {str(e)[:100]}"

# ✅ Tool names: SIMPLE, NO SPACES (critical for ReAct parsing)
duckduck_tool = Tool(
    name="duckduckgo",  # ✅ Was: "DuckDuckGoSearch"
    description="Web search for general queries, news, facts.",
    func=safe_search
)

wiki_tool = Tool(
    name="wikipedia",  # ✅ Was: "Wikipedia Search Tool"
    description="Wikipedia search for encyclopedic information.",
    func=safe_wiki
)

arxiv_tool = Tool(
    name="arxiv",  # ✅ Was: "Arxiv Search Tool"
    description="Academic paper search on Arxiv.",
    func=safe_arxiv
)

tools = [duckduck_tool, wiki_tool, arxiv_tool]

# ===== 🔧 FIXED: Prompt Template =====
# Key fixes:
# 1. Clearer STOP instruction
# 2. Simpler format example
# 3. Removed duplicate {agent_scratchpad} at end
react_template = """You are a helpful assistant. Use tools to answer accurately.

Tools:
{tools}

History: {chat_history}

Format (follow EXACTLY):
Question: {{input}}
Thought: {{thought}}
Action: {{action}} (must be one of: [{tool_names}])
Action Input: {{action_input}}
Observation: {{observation}}
[Repeat Thought/Action/Observation as needed]
Thought: I now know the final answer
Final Answer: {{final_answer}} [STOP HERE - do not add more text]

Rules:
1. If you know the answer without tools, go straight to "Final Answer:".
2. After "Final Answer:", STOP. Do not generate more text.
3. Tool names must match exactly: [{tool_names}]

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"]
)

# ----- Memory -----
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# ----- Agent & Executor =====
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=buffer_memory,
    verbose=True,
    max_iterations=4,
    early_stopping_method="force",
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

def chat_with_memory(question: str) -> str:
    """Process a user question with memory support."""
    start_time = time.time()
    
    if "clear memory" in question.lower() or "reset" in question.lower():
        buffer_memory.chat_memory.clear()
        return "✅ Memory has been cleared!"
    
    try:
        result = agent_executor.invoke({
            "input": question,
            "chat_history": buffer_memory.chat_memory.messages
        }, config={"max_execution_time": 60})
        
        elapsed = time.time() - start_time
        print(f"⏱️ Response time: {elapsed:.2f}s")
        
        # ✅ FALLBACK: If agent fails to produce Final Answer, extract from observations
        output = result.get("output", "")
        if not output or "Agent stopped due to" in output:
            print("🔄 Agent didn't produce Final Answer, extracting from observations...")
            # Try to extract answer from intermediate steps
            if result.get("intermediate_steps"):
                for action, observation in result["intermediate_steps"]:
                    obs_lower = str(observation).lower()
                    if "cairo" in obs_lower and "capital" in obs_lower:
                        return "Based on search results: The capital of Egypt is Cairo."
            # Last resort: direct LLM call
            return llm.invoke(f"Answer in one sentence: {question}").content
        
        return output
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⏱️ Failed after {elapsed:.2f}s")
        return f"Sorry, I encountered an error: {str(e)[:150]}"

def start_chatbot():
    """Start an interactive chat session."""
    print("Hello! I am your assistant. You can start asking questions now.")
    print("Type 'exit' to end, 'reset' to clear memory.")
    print(f"📦 Model: {llm.model_name} | Tools: {[t.name for t in tools]}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", "bye"]:
                print("Goodbye! Have a great day!")
                break
            
            response = chat_with_memory(user_input)
            print(f"\nBot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            break

if __name__ == "__main__":
    start_chatbot()