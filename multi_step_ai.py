from typing import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END

load_dotenv()
#gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Define State
class AIState(TypedDict):
    question: str
    topic: str
    explanation: str
    final_answer: str


# Node 1 Analyze the question
def analyze_question(state: AIState):

    question = state["question"]

    return {
        "topic": question
    }


# Node 2  explanation using Gemini
def generate_explanation(state: AIState):

    response = model.invoke([
        SystemMessage(
            content="You are a helpful AI teacher. Explain concepts in simple English with examples."
        ),
        HumanMessage(
            content=f"Explain this question: {state['topic']}"
        )
    ])

    answer = response.content

    # Handle text-block responses if returned as a list
    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return {
        "explanation": answer
    }


# Node 3 Format the answer
def format_response(state: AIState):

    final = f""" Your Question: {state['question']}AI Explanation:
    {state['explanation']}
    """

    return {
        "final_answer": final
    }


# Build graph
builder = StateGraph(AIState)
# add nodes
builder.add_node("analyze", analyze_question)
builder.add_node("generate", generate_explanation)
builder.add_node("format", format_response)

# Connecting nodes
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "generate")
builder.add_edge("generate", "format")
builder.add_edge("format", END)

# Compile
graph = builder.compile()


# Run
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke({
        "question": user_input,
        "topic": "",
        "explanation": "",
        "final_answer": ""
    })

    print("\n", result["final_answer"])
