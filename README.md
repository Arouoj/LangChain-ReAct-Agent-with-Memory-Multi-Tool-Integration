# LangChain-ReAct-Agent-with-Memory-Multi-Tool-Integration

A conversational AI assistant built with LangChain, Groq's Mixtral model, and multiple tools (web search, Wikipedia, ArXiv). The agent uses the ReAct (Reasoning + Acting) framework to break down questions and fetch information step by step, while maintaining conversation history for context.

## Features
- **Multi-tool support**: DuckDuckGo search, Wikipedia, and ArXiv academic search.
- **Persistent conversation memory**: Remembers previous exchanges (cleared with "clear memory" or "reset").
- **Powered by Groq's Mixtral 8x7B**: Fast and high-quality inference.
- **ReAct prompting**: The agent explicitly plans and executes tool calls.
- **Easy to run**: Simple CLI interface.

## Prerequisites
- Python 3.8 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-chatbot-agent.git
   cd langchain-chatbot-agent