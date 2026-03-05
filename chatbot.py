import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

# Initialize the LLM (Mixtral via Groq)
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# ----- Tools -----
duckduck = DuckDuckGoSearchResults()
duckduck_tool = Tool(
    name="DuckDuckGoSearch",
    description="A web search engine. Use this as a search engine for general queries.",
    func=duckduck.run
)

wiki_wrapper = WikipediaAPIWrapper()
wiki_query = WikipediaQueryRun(api_wrapper=wiki_wrapper)
wiki_tool = Tool(
    name="Wikipedia Search Tool",
    description="An API search tool, use it to retrieve information from Wikipedia articles.",
    func=wiki_query.run
)

arxiv_wrapper = ArxivAPIWrapper()
arxiv_query = ArxivQueryRun(api_wrapper=arxiv_wrapper)
arxiv_tool = Tool(
    name="Arxiv Search Tool",
    description="An API search tool, use it to search for academic papers, preprints, and research articles on Arxiv.",
    func=arxiv_query.run
)

tools = [duckduck_tool, wiki_tool, arxiv_tool]

# ----- Prompt Template -----
react_template = """
<s>[INST] You are an AI assistant that answers questions by breaking them down and using tools precisely.
Answer the following questions as best you can. You have access to the following tools:
{tools}

Previous conversation history:
{chat_history}

Use the following format STRICTLY and EXACTLY:
Question: the input question you must answer  
Thought: you should always think about what to do  
Action: the action to take, should be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation can repeat N times)  
Thought: I now know the final answer  
Final Answer: the final answer to the original input question  

IMPORTANT: After providing the Final Answer, you must STOP. Do not generate any additional text or steps.
IMPORTANT: If a tool result is bad, try another tool.
IMPORTANT: Use the chat history to provide more contextual and consistent responses.

Question: {input}
{agent_scratchpad} [/INST]
Thought: Let me break this down step by step and follow the format exactly...
{agent_scratchpad}</s>
"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"]
)

# ----- Memory -----
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# ----- Agent & Executor -----
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=buffer_memory,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

def chat_with_memory(question: str) -> str:
    """Process a user question with memory support."""
    if "clear memory" in question.lower() or "reset" in question.lower():
        buffer_memory.chat_memory.clear()
        return "Memory has been cleared!"
    
    result = agent_executor.invoke({
        "input": question,
        "chat_history": buffer_memory.chat_memory.messages
    })
    return result["output"]

def start_chatbot():
    """Start an interactive chat session."""
    print("Hello! I am your assistant. You can start asking questions now.\n(Type 'exit' to end the chat.)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye! Have a great day!")
            break
        
        response = chat_with_memory(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    start_chatbot()