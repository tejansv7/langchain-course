"""
Agent Loop using Native Google Cloud (GCP) Vertex AI Function Calling
====================================================================
This script demonstrates how raw function calling and manual agent loops work
using Google's official native GenAI SDK (`google-genai`), WITHOUT LangChain
and WITHOUT attempting to imitate Ollama's data structures.

Comparison with Ollama:
- Ollama uses: `ollama.chat(model=..., tools=[...], messages=[...])`
- GCP Native uses: `client.models.generate_content(model=..., contents=[...], config=...)`
- Tool results in Ollama: `{"role": "tool", "content": "..."}`
- Tool results in GCP: `types.Part.from_function_response(name=..., response={"result": ...})`
"""

import sys
import os

# Fix Windows console UTF-8 encoding for emojis/special characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

# Official Google GenAI SDK (Vertex AI)
from google import genai
from google.genai import types
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemini-2.5-flash"

# Initialize native GCP Vertex AI Client
client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
)


# ==============================================================================
# 1. Local Tool Functions (Pure Python)
# ==============================================================================


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
    return round(float(price) * (1 - discount / 100), 2)


tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ==============================================================================
# 2. Native GCP Tool Declarations (FunctionDeclaration & Schema)
# ==============================================================================
# In GCP, raw function schemas are declared using types.FunctionDeclaration.
# Notice how clean and strongly typed this is compared to raw JSON strings.

tool_declarations = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_product_price",
            description="Look up the price of a product in the catalog.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "product": types.Schema(
                        type="STRING",
                        description="The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    )
                },
                required=["product"],
            ),
        ),
        types.FunctionDeclaration(
            name="apply_discount",
            description="Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "price": types.Schema(
                        type="NUMBER",
                        description="The original price of the product",
                    ),
                    "discount_tier": types.Schema(
                        type="STRING",
                        description="The discount tier: 'bronze', 'silver', or 'gold'",
                    ),
                },
                required=["price", "discount_tier"],
            ),
        ),
    ]
)

# Configuration with system instruction, tools, and deterministic temperature
config = types.GenerateContentConfig(
    system_instruction=(
        "You are a helpful shopping assistant. "
        "You have access to a product catalog tool and a discount tool.\n\n"
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
    tools=[tool_declarations],
    temperature=0,
)


# ==============================================================================
# 3. Traced Native GCP Model Call
# ==============================================================================


@traceable(name="GCP Gemini Raw Chat", run_type="llm")
def gemini_generate_traced(contents):
    return client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )


# ==============================================================================
# 4. Native GCP Agent Loop
# ==============================================================================


@traceable(name="GCP Native Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # In GCP GenAI SDK, conversations are tracked as a list of types.Content objects:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 1. Send conversation history to Gemini
        response = gemini_generate_traced(contents)

        # 2. Check if the model wants to call tools
        function_calls = response.function_calls

        # If no function calls requested, this is the final answer!
        if not function_calls:
            print(f"\nFinal Answer: {response.text}")
            return response.text

        # 3. Process the first requested tool call
        tool_call = function_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.args  # Directly a Python dictionary in GCP!

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # 4. Execute the Python tool locally
        observation = tool_to_use(**tool_args)
        print(f"  [Tool Result] {observation}")

        # 5. Append model's response to history (contains the FunctionCall part)
        contents.append(response.candidates[0].content)

        # 6. Append tool output to history using native GCP FunctionResponse part
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation},
                    )
                ],
            )
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello GCP Native Function Calling Agent!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
