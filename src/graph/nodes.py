from src.graph.state import ChatbotState
from src.rag.retriever import search
from src.rag.generator import (
    get_llm,
    format_context
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

def router_node(state: ChatbotState):
    """
    Use the local LLM to classify the user's intent.

    Routes:
    - conversation
    - rag
    """

    question = state["question"]

    llm = get_llm()

    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the intent router for an e-commerce chatbot.

Classify the user's message into exactly ONE of these routes:

conversation:
Use for greetings, thanks, casual conversation, or messages
that do not require product or company-policy information.

rag:
Use for questions about products, product specifications,
prices, availability, recommendations, returns, refunds,
delivery, cancellation, warranty, payment, or company policies.

Important:
If the message contains both a greeting and a product or
policy question, classify it as rag.

Return ONLY one word:
conversation
or
rag
""".strip()
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    chain = router_prompt | llm

    response = chain.invoke(
        {
            "question": question
        }
    )

    route = response.content.strip().lower()

    # Safe fallback
    if route not in {"conversation", "rag"}:
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

    if "thank" in question or "thanks" in question:
        answer = (
            "You're welcome! Let me know if you have any other "
            "product or policy questions."
        )

    elif any(
        greeting in question
        for greeting in [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening"
        ]
    ):
        answer = (
            "Hello! How can I help you with our products "
            "or policies today?"
        )

    else:
        answer = "Hello! How can I help you today?"

    return {
        "answer": answer,
        "messages": [
        AIMessage(content=answer)
    ]
    }


def retrieve_node(state: ChatbotState):
    """
    Retrieve relevant documents from Chroma.
    """

    retrieval_query = state["retrieval_query"]

    documents = search(
        retrieval_query,
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
        "answer": response.content,
        "messages": [
        AIMessage(content=response.content)
        ]
    }

##### rewrite node to re write customer follow up question####
def rewrite_query_node(state: ChatbotState):
    """
    Rewrite a follow-up question into a standalone retrieval query
    using the conversation history.
    """

    question = state["question"]
    messages = state["messages"]

    llm = get_llm()

    # Exclude the latest HumanMessage because the current question
    # is already passed separately below.
    previous_messages = messages[:-1]

    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in previous_messages
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You rewrite e-commerce customer questions for document retrieval.

Use the conversation history only to resolve references in the
latest question, such as:

- it
- they
- that product
- the first one
- the second one
- the third one
- the laptop you recommended
- that policy

Create a standalone search query containing the relevant product,
product name, or topic from the conversation history.

Do not answer the customer's question.
Do not invent information.
Do not add information that is not present in the conversation.

If the latest question is already standalone and understandable
without the conversation history, return it unchanged.

Return ONLY the rewritten search query.
""".strip()
            ),
            (
                "human",
                """
Conversation history:
{history}

Latest question:
{question}

Standalone retrieval query:
""".strip()
            )
        ]
    )

    chain = rewrite_prompt | llm

    response = chain.invoke(
        {
            "history": history,
            "question": question
        }
    )

    retrieval_query = response.content.strip()

    print(f"ORIGINAL QUERY: {question}")
    print(f"RETRIEVAL QUERY: {retrieval_query}")

    return {
        "retrieval_query": retrieval_query
    }