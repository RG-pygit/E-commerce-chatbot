from src.graph.state import ChatbotState
from src.rag.retriever import (
    search,
    find_exact_product_names,
    search_exact_product,
    search_exact_products
)
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

    retrieval_query = state["retrieval_query"]

    exact_product_names = find_exact_product_names(
        retrieval_query
    )

    if len(exact_product_names) > 1:

        documents = search_exact_products(
            exact_product_names
        )

        print(
            "RETRIEVAL MODE: multiple exact products -> "
            f"{exact_product_names}"
        )

    elif len(exact_product_names) == 1:

        product_name = exact_product_names[0]

        documents = search_exact_product(
            product_name
        )

        print(
            "RETRIEVAL MODE: exact product -> "
            f"{product_name}"
        )

    else:

        documents = search(
            retrieval_query,
            k=3
        )

        print("RETRIEVAL MODE: semantic search")

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

12. If the customer refers to a specific product using an incomplete
or ambiguous product name, and you cannot reliably determine which
exact product they mean, do not guess, substitute a similar product,
or claim that the product does not exist.

In that case, ask:

"Please provide the full product name."

If multiple specific products are ambiguous, ask:

"Please provide the full product names."

Do NOT ask for full product names when the customer is making a
general request such as asking for product recommendations,
browsing products, or asking which product they should buy.
For general requests, answer using the relevant products in the
retrieved context.

13. If the specific information needed to answer the question is
not available in the retrieved context, respond:

"I don't have enough information in the available product and
policy data."

If the specific information needed to answer the question is not available in the retrieved context, respond:

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
You rewrite e-commerce customer questions into standalone
queries for document retrieval.

Your ONLY job is to resolve references using the conversation
history.

Rules:

1. Do NOT answer the customer's question.

2. Do NOT decide which product is cheapest, most expensive,
   has the most RAM, most storage, longest battery life,
   longest warranty, or is otherwise the winner of a comparison.

3. For a question referring to ONE product, replace the
   reference with the exact product name from the conversation.

Example:

Conversation:
Assistant recommended FitPulse Smartwatch 194.

Question:
How much does that smartwatch cost?

Output:
FitPulse Smartwatch 194 price


4. For a question referring to MULTIPLE products using words
   such as "those", "these", "them", "the options", or similar
   references, include ALL relevant product names from the
   conversation history.

Example:

Conversation:
Assistant discussed Product A, Product B, and Product C.

Question:
Which of those has the most RAM?

Output:
Compare RAM of Product A, Product B, and Product C


5. Never replace a multi-product comparison with only one
   product name.

6. For positional references such as "first", "second", or
   "third", resolve the positions using the order in which
   products were presented in the conversation.

7. Do not invent product names or information.

8. If the latest question is already standalone and does not
   depend on conversation history, return a concise standalone
   retrieval query.

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