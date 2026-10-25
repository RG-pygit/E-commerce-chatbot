from langgraph.graph import StateGraph, START, END

from src.graph.state import ChatbotState
from src.graph.nodes import (router_node, conversation_node,rewrite_query_node,retrieve_node, generate_node)

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

def decide_route(state: ChatbotState):
    """
    Decide which branch LangGraph should follow
    after the router node.
    """

    return state["route"]


def build_graph():

    # Create graph using our shared state
    builder = StateGraph(ChatbotState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("conversation", conversation_node)
    builder.add_node("rewrite_query",rewrite_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Connect nodes
    # Start with router
    builder.add_edge(START, "router")

    # Conditional routing
    builder.add_conditional_edges(
    "router",
    decide_route,
    {
        "conversation": "conversation",
        "rag": "rewrite_query"
    }
        )

    # Conversation path
    builder.add_edge("conversation", END)

    # RAG path
    builder.add_edge("rewrite_query","retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    

    # Compile graph
    memory = InMemorySaver()

    return builder.compile(
    checkpointer=memory
)


if __name__ == "__main__":

    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "customer-1"
        }
    }
   
    while True:

        question = input(
            "\nAsk a product or policy question "
            "(or type 'exit'): "
        )

        if question.strip().lower() == "exit":
            print("Chat ended.")
            break

        input_state = {
            "question": question,
            "messages": [
                HumanMessage(content=question)
            ]
        }

        result = graph.invoke(
            input_state,
            config=config
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])

        print("\n" + "=" * 60)
        print("MESSAGES STORED")
        print("=" * 60)

        for message in result["messages"]:
            print(
                f"{type(message).__name__}: "
                f"{message.content}"
            )

    

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        for document in result["documents"]:

            source = document.metadata.get(
            "source",
            "Unknown"
        )

            item = (
            document.metadata.get("product_name")
            or document.metadata.get("policy_type")
            or "Unknown"
        )

            print(f"- {source}: {item}")