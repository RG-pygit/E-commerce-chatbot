from langgraph.graph import StateGraph, START, END

from src.graph.state import ChatbotState
from src.graph.nodes import (router_node, conversation_node,rewrite_query_node,product_retrieve_node,
policy_retrieve_node,product_generate_node, policy_generate_node,generate_node,product_agent_node,policy_agent_node)

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

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
    builder.add_node("product_retrieve",product_retrieve_node)
    builder.add_node("policy_retrieve",policy_retrieve_node)
    builder.add_node("product_generate",product_generate_node) 
    builder.add_node("policy_generate",policy_generate_node)
    builder.add_node("generate", generate_node)
    builder.add_node("product_agent",product_agent_node)
    builder.add_node("policy_agent",policy_agent_node)

    # Connect nodes
    # Start with router
    builder.add_edge(START, "router")

    # Conditional routing
    builder.add_conditional_edges(
    "router",
    decide_route,
    {
        "conversation": "conversation",
        "product": "rewrite_query",
        "policy": "rewrite_query"
    }
)

    # Conversation path
    builder.add_edge("conversation", END)

    # RAG path
    builder.add_conditional_edges(
    "rewrite_query",
    decide_route,
    {
        "product": "product_agent",
        "policy": "policy_agent"
    }
)
    builder.add_edge("product_agent","product_generate")
    builder.add_edge("policy_agent", "policy_generate")
    builder.add_edge("product_generate", END)
    builder.add_edge("policy_generate", END)
    

    # # Compile graph
    # memory = InMemorySaver()

    # return builder.compile(
    # checkpointer=memory

    from pathlib import Path

    db_path = Path("chatbot_memory.db").resolve()

    print("SQLITE DATABASE PATH:", db_path)

    connection = sqlite3.connect(
    db_path,
    check_same_thread=False
    )

    memory = SqliteSaver(connection)

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
        print("SOURCES")
        print("=" * 60)

        for document in result.get("documents", []):

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