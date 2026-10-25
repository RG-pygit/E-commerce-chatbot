from langgraph.graph import StateGraph, START, END

from src.graph.state import ChatbotState
from src.graph.nodes import (router_node, conversation_node,retrieve_node, generate_node)

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
                "rag": "retrieve"
            }
        )

    # Conversation path
    builder.add_edge("conversation", END)

    # RAG path
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    

    # Compile graph
    graph = builder.compile()

    return graph


if __name__ == "__main__":

    graph = build_graph()

    question = input("\nAsk a product or policy question: ")

    initial_state = {
    "question": question,
    "route": "",
    "documents": [],
    "answer": ""
}

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(result["answer"])

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