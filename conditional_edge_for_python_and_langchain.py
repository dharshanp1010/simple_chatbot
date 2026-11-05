from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class ChatState(TypedDict):
    message: str
    response: str


#  Decide the question category
def classify_question(state: ChatState):

    message = state["message"].lower()

    if "python" in message:
        return "python"

    elif "langgraph" in message:
        return "langgraph"

    else:
        return "other"


#  Python node
def python_node(state: ChatState):
    return {
        "response": "Your question is related to Python."
    }


#  LangGraph node
def langgraph_node(state: ChatState):
    return {
        "response": "Your question is related to LangGraph."
    }


# if Other questions
def other_node(state: ChatState):
    return {
        "response": "Sorry, I can only help with Python and LangGraph."
    }


# Builder
builder = StateGraph(ChatState)

builder.add_node("python_node", python_node)
builder.add_node("langgraph_node", langgraph_node)
builder.add_node("other_node", other_node)

builder.add_conditional_edges(
    START,
    classify_question,
    {
        "python": "python_node",
        "langgraph": "langgraph_node",
        "other": "other_node"
    }
)

builder.add_edge("python_node", END)
builder.add_edge("langgraph_node", END)
builder.add_edge("other_node", END)

graph = builder.compile()


# Run chatbot
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke({
        "message": user_input,
        "response": ""
    })

    print("Bot:", result["response"])
