from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.3
)
# Chatbot node
def chatbot(state: MessagesState):
    response = model.invoke(
        [
            SystemMessage(
                content="You are a chatbot. Remember the conversation context."
            )
        ] + state["messages"]
    )

    return {"messages": [response]}

# Build graph
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Add checkpointer
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# checkpointer saves the graph state so it can be reused so thread id is required
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}
# Chat loop
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)]
        },
        config=config
    )

    answer = result["messages"][-1].content

    # Handle Gemini responses returned as text blocks
    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )

    print("Gemini:", answer)
