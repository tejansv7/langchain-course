import sys
import os 
from typing import List

# Fix Windows console UTF-8 encoding for emojis/special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# --- LLM Imports ---
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Tool Imports ---
from langchain_tavily import TavilySearch
from ddgs import DDGS


# 1. DuckDuckGo Search Tool (Free, no API key required)
@tool
def duckduckgo_search(query: str) -> str:
    """Search the web for information using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return str(results)


# 2. Tavily Search Tool (AI-optimized agent search)
tavily_search = TavilySearch(max_results=3)


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


# --- Model Initialization ---
# llm = ChatOpenAI(model="gpt-5")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
)

# --- Tool Configuration ---
# tools = [TavilySearch()]                  # Original single Tavily tool
# tools = [duckduckgo_search]               # Free DuckDuckGo tool only
tools = [tavily_search, duckduckgo_search]  # Dual search tools (Tavily + DuckDuckGo)

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    # --- Prompt Options ---
    # Original prompt (single query):
    #original_query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"

    # Multi-tool comparison prompt:
    dual_search_query = (
        "Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn. "
        "Use BOTH the Tavily search tool and the DuckDuckGo search tool so we can compare findings. "
        "In your answer, clearly list the findings from Tavily and the findings from DuckDuckGo, "
        "and include the source URLs for each."
    )

    result = agent.invoke(
        {
            "messages": HumanMessage(content=dual_search_query)
            #"messages": HumanMessage(content=original_query)  # Uncomment for original prompt
        }
    )

    # --- Output Printing ---
    # print(result)  # Original: raw dictionary dump with message history & tokens

    print("\n=== Agent Response ===")
    if "structured_response" in result:
        structured: AgentResponse = result["structured_response"]
        print(structured.answer)
        if structured.sources:
            print("\nSources:")
            for s in structured.sources:
                print(f"- {s.url}")
    else:
        print(result)


if __name__ == "__main__":
    main()
