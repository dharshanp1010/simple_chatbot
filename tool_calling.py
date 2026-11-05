from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Create tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number."""
    if b == 0:
        return "Error: Cannot divide by zero."

    return str(a / b)


# 3. Collect all tools
tools = [add, subtract, multiply, divide]

# Connect tools to Gemini
model_with_tools = model.bind_tools(tools)


# chatbot node
def chatbot(state: MessagesState):

    response = model_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


# builder
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))


builder.add_edge(START, "chatbot")

# Decide whether to execute a tool or finish
builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

# After tool execution, return to Gemini
builder.add_edge("tools", "chatbot")

# Compile
graph = builder.compile()


# 7. Run chatbot
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    })

    answer = result["messages"][-1].content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )

    print("\nGemini:", answer)
