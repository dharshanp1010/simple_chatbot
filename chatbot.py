from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#create the state
class ChatState(TypedDict):
    user_message: str
    bot_response: str

#create the node
def chatbot(state: ChatState):

    message = state["user_message"].lower()

    if "hello" in message or "hi" in message:
        response = "Hello! How can I help you?"

    elif "name" in message:
        response = "I am a LangGraph chatbot."

    elif "bye" in message:
        response = "Goodbye! Have a great day."

    else:
        response = "Sorry, I don't understand your question."

    return {
        "bot_response": response
    }

#builder
builder = StateGraph(ChatState)

#add the node
builder.add_node("chatbot", chatbot)

#connect the edge
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

#compile
graph = builder.compile()

while True:
    #get user input for chatbot
    user_input = input("You: ")
    
    # if exit is input
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    #invoke the chat bot
    result = graph.invoke({
        "user_message": user_input,
        "bot_response": ""
    })

    print("Chatbot:", result["bot_response"])
