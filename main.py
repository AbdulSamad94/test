# ############################# AI Chatbot Agent Integrate with Next JS Website #############################

import os
import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    RunConfig,
    RunContextWrapper,
)
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Provider
external_provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_provider,
)

run_config = RunConfig(
    model=model,
    model_provider=external_provider,
    tracing_disabled=True,
)


@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the AI Chatbot Agent!").send()


@function_tool
def get_updated_products(query: str) -> str:
    """
    Function to get the latest product updates.
    """
    print(f"Received query: {query}")
    return f"The latest product updates include new features, improved performance, and enhanced user experience across various categories."


@function_tool
def get_updated_website(query: str) -> str:
    """
    Function to get the latest website updates.
    """
    print(f"Received query: {query}")
    return f"The latest website updates include a new design, improved navigation, and enhanced content to provide a better user experience."


@function_tool
def get_update_payment(query: str) -> str:
    """
    Function to get the latest payment updates.
    """
    print(f"Received query: {query}")
    return f"The latest payment updates include new payment methods, enhanced security features, and improved transaction processing times."


# Products Agent
products_agent = Agent(
    name="Products_Agent",
    instructions="You are a specialized agent for product-related all queries. You can provide information about products, their features, price of each product and specifications.",
    model=model,
    tools=[get_updated_products],  # Register the function tool and product info tool
)

# Website Overview Agent
website_overview_agent = Agent(
    name="Website_Overview_Agent",
    instructions="You are a specialized agent for website overview queries. You can provide all website information, all overview and details.",
    model=model,
    tools=[get_updated_website],  # Register the function tool
)

# Payment Agent
payment_agent = Agent(
    name="Payment_Agent",
    instructions="You are a specialized agent for payment-related queries. You can provide information about payment methods, payment status, and any payment-related issues.",
    model=model,
    tools=[get_update_payment],  # Register the function tool
)


# AI Chatbot Orschestrator Agent
ai_chatbot_agent = Agent(
    name="AI_Chatbot_Agent",
    instructions="You are a help full assistant of AI Chatbot, You have a tools and agents to solve user queries",
    model=model,
    tools=[
        products_agent.as_tool(
            tool_name="Products_Agent",
            tool_description="You are a specialized agent for product-related queries. You can provide information about products, their features, price of each product and specifications.",
        ),
        website_overview_agent.as_tool(
            tool_name="Website_Overview_Agent",
            tool_description="You are a specialized agent for website overview queries. You can provide all website information, all overview and details.",
        ),
        payment_agent.as_tool(
            tool_name="Payment_Agent",
            tool_description="You are a specialized agent for payment-related queries. You can provide information about payment methods, payment status, and any payment-related issues.",
        ),
    ],
)


@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(ai_chatbot_agent, message.content, run_config=run_config)
    await cl.Message(content=f"Response: {result.final_output}").send()
    print(f"Response:{result.final_output}")
