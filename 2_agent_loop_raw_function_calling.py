import sys
import os
from types import SimpleNamespace

# Fix Windows console UTF-8 encoding for emojis/special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

# --- Ollama Setup (Original) ---
# import ollama
# MODEL = "qwen3:1.7b"

# --- GCP Vertex AI Setup ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

MODEL = "gemini-2.5-flash"
from langsmith import traceable

MAX_ITERATIONS = 10


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Chat call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

# Original Ollama Chat Helper:
# @traceable(name="Ollama Chat", run_type="llm")
# def ollama_chat_traced(messages):
#     return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# GCP Vertex AI Alternative:
llm_gcp = ChatGoogleGenerativeAI(
    model=MODEL,
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
    temperature=0,
)
llm_with_tools = llm_gcp.bind_tools(tools_for_llm)


@traceable(name="GCP Gemini Chat", run_type="llm")
def gcp_chat_traced(messages):
    lc_messages = []
    last_call_id = "call_0"
    for m in messages:
        if isinstance(m, dict):
            r = m.get("role")
            c = m.get("content", "")
            if r == "system":
                lc_messages.append(SystemMessage(content=c))
            elif r == "user":
                lc_messages.append(HumanMessage(content=c))
            elif r == "tool":
                lc_messages.append(ToolMessage(content=c, tool_call_id=last_call_id))
        elif hasattr(m, "raw_ai_message"):
            raw = m.raw_ai_message
            if raw.tool_calls:
                last_call_id = raw.tool_calls[0].get("id", "call_0")
            lc_messages.append(raw)
        else:
            lc_messages.append(m)

    res = llm_with_tools.invoke(lc_messages)
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=tc["name"], arguments=tc["args"]))
        for tc in res.tool_calls
    ]
    txt = res.content
    if isinstance(txt, list) and len(txt) > 0 and isinstance(txt[0], dict):
        txt = txt[0].get("text", "")
    msg = SimpleNamespace(content=txt, tool_calls=calls, raw_ai_message=res)
    return SimpleNamespace(message=msg)


# --- Agent Loop ---


@traceable(name="Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Original Ollama invocation:
        # response = ollama_chat_traced(messages=messages)

        # GCP Vertex AI alternative:
        response = gcp_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)


        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
