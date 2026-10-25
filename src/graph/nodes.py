from src.graph.state import ChatbotState
from src.rag.retriever import search
from src.rag.generator import (
    get_llm,
    format_context
)

from langchain_core.prompts import ChatPromptTemplate

def router_node(state: ChatbotState):
    """
    Decide whether the user's message is conversational
    or should go through the RAG pipeline.
    """

    question = state["question"].strip().lower()

    conversational_messages = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you"
    }

    if question in conversational_messages:
        route = "conversation"
    else:
        route = "rag"
    print(f"ROUTE SELECTED: {route}")
    return {
        "route": route
    }

def conversation_node(state: ChatbotState):
    """
    Handle simple conversational messages without
    searching the vector database.
    """

    question = state["question"].strip().lower()

    if question in {"hi", "hello", "hey", "hi there", "hello there"}:
        answer = "Hello! How can I help you with our products or policies today?"

    elif question in {"good morning", "good afternoon", "good evening"}:
        answer = "Hello! How can I help you with our products or policies today?"

    elif question in {"thanks", "thank you"}:
        answer = "You're welcome! Let me know if you have any other product or policy questions."

    else:
        answer = "Hello! How can I help you today?"

    return {
        "answer": answer
    }


def retrieve_node(state: ChatbotState):
    """
    Retrieve relevant documents from Chroma.
    """

    question = state["question"]

    documents = search(
        question,
        k=3
    )

    return {
        "documents": documents
    }


def generate_node(state: ChatbotState):
    """
    Generate an answer using the documents
    retrieved by the retrieve node.
    """
    print(">>> LANGGRAPH GENERATE NODE IS RUNNING <<<")
    question = state["question"]
    documents = state["documents"]

    context = format_context(documents)

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an e-commerce customer support assistant.

Answer the customer's question using ONLY the retrieved context.

Follow these rules:

1. Read the customer's question carefully and identify exactly what information is being requested.

2. Read the retrieved context carefully and identify only the information relevant to that specific request.

3. Use only facts explicitly provided in the retrieved context.

4. Do not answer with unrelated information simply because it is available in the retrieved context.

5. You may perform simple logical reasoning, comparisons, and calculations when they are directly supported by the retrieved context.

6. When comparing values such as price, quantity, time, storage, RAM, battery life, ratings, or other numerical values, reason about them carefully before answering.

7. Do not invent product specifications, availability, prices, policies, conditions, or other information.

8. Do not use outside knowledge.

9. If multiple retrieved items are relevant to the customer's request, present the relevant options clearly.

10. Include important conditions, limitations, or exceptions from the retrieved context when they affect the answer.

11. If the retrieved context contains information about the product or topic, but does NOT contain the specific information requested by the customer, do not substitute another product attribute or unrelated fact.

12. If the specific information needed to answer the question is not available in the retrieved context, respond:

"I don't have enough information in the available product and policy data."

Give a direct, concise, customer-friendly answer.
""".strip()
        ),
        (
            "human",
            """
Customer question:
{question}

Retrieved context:
{context}

Answer the customer's specific question using only the retrieved context.
""".strip()
        )
    ]
)

    llm = get_llm()

    chain = prompt | llm

    #testing######

    messages = prompt.format_messages(
    question=question,
    context=context
    )

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    return {
        "answer": response.content
    }