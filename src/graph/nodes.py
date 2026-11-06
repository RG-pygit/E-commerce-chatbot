from src.graph.state import ChatbotState
from src.rag.retriever import (
    search,
    find_exact_product_names,
    search_exact_product,
    search_exact_products,
    search_by_source_type
)
from src.rag.generator import (
    get_llm,
    format_context
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.agents.product_agent import run_product_agent
from src.agents.policy_agent import run_policy_agent

def router_node(state: ChatbotState):
    """
    Use the local LLM to classify the user's intent.

    Routes:
    - conversation
    - product
    - policy
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

product:
Use when the customer is asking about products themselves.

This includes:
- product recommendations
- product specifications
- prices
- availability
- RAM
- storage
- battery
- warranty of a specific product
- comparing products
- follow-up questions about previously discussed products

Do NOT classify delivery, shipping, returns, refunds,
order cancellation, payment rules, or other company policies
as product.

policy:
Use when the customer is asking about company rules, processes,
or policies that apply to purchases or orders.

This includes:
- delivery or shipping time
- delivery or shipping rules
- returns
- refunds
- cancellation
- payment policies
- warranty policies
- other company rules or policies

Examples:

"How long does delivery take?" -> policy

"When will my order be delivered?" -> policy

"Can I return a product after 20 days?" -> policy

"Can I cancel an order after it has shipped?" -> policy

"What is the refund policy?" -> policy

Important:

If a message contains a greeting plus a product question,
classify it as product.

If a message contains a greeting plus a policy question,
classify it as policy.

Return ONLY one word:

conversation
product
policy
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
    if route not in {
        "conversation",
        "product",
        "policy"
    }:
        route = "product"

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

def product_agent_node(state: ChatbotState):
    """
    Use the Product Agent to choose and execute
    the appropriate product retrieval tool.
    """

    retrieval_query = state["retrieval_query"]

    print(
        f"PRODUCT AGENT INPUT: {retrieval_query}"
    )

    documents = run_product_agent(
        retrieval_query
    )

    active_product_names = [
        document.metadata.get("product_name")
        for document in documents
        if document.metadata.get("product_name")
    ]

    print(
        "ACTIVE PRODUCTS:",
        active_product_names
    )

    return {
        "documents": documents,
        "active_product_names": active_product_names
    }

def policy_agent_node(state: ChatbotState):
    """
    Use the Policy Agent to retrieve relevant
    company policy documents.
    """

    retrieval_query = state["retrieval_query"]

    print(
        f"POLICY AGENT INPUT: {retrieval_query}"
    )

    documents = run_policy_agent(
        retrieval_query
    )

    return {
        "documents": documents,
        "active_product_names": []
    }

def product_retrieve_node(state: ChatbotState):

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

        documents = search_by_source_type(
            query=retrieval_query,
            source_type="product",
            k=3
        )

        print("RETRIEVAL MODE: product semantic search")

    active_product_names = [
        document.metadata.get("product_name")
        for document in documents
        if document.metadata.get("product_name")
    ]

    print(
        "ACTIVE PRODUCTS:",
        active_product_names
    )

    return {
        "documents": documents,
        "active_product_names": active_product_names
    }

def policy_retrieve_node(state: ChatbotState):

    retrieval_query = state["retrieval_query"]

    documents = search_by_source_type(
        query=retrieval_query,
        source_type="policy",
        k=3
    )

    print("RETRIEVAL MODE: policy semantic search")

    return {
        "documents": documents,
        "active_product_names": []
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

13. If a requested attribute has the value "N/A", "Unknown",
"Not available", or an empty value, treat that attribute as
unavailable information.

Do not interpret an unavailable value as meaning that the product
has the feature, and do not infer or guess a value.

If the requested attribute is unavailable for all relevant products,
do not select a winner for the comparison.

14. If the specific information needed to answer the question is
not available in the retrieved context, respond:

"I don't have enough information in the available product and
policy data."

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

def product_generate_node(state: ChatbotState):
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
You are an e-commerce product assistant.

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

13. If a requested attribute has the value "N/A", "Unknown",
"Not available", or an empty value, treat that attribute as
unavailable information.

Do not interpret an unavailable value as meaning that the product
has the feature, and do not infer or guess a value.

If the requested attribute is unavailable for all relevant products,
do not select a winner for the comparison.

14. If the specific information needed to answer the question is
not available in the retrieved context, respond:

"I don't have enough information in the available product data."

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

    print("QUESTION:")
    print(question)

    print("\nCONTEXT LENGTH:")
    print(len(context))

    print("\nCONTEXT:")
    print(context)
    
    print(">>> BEFORE PRODUCT LLM CALL <<<")

    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    print(">>> AFTER PRODUCT LLM CALL <<<")

    return {
        "answer": response.content,
        "messages": [
        AIMessage(content=response.content)
        ]
    }

def policy_generate_node(state: ChatbotState):
    """
    Generate an answer for policy-related questions
    using retrieved policy documents.
    """

    question = state["question"]
    documents = state["documents"]

    context = format_context(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an e-commerce customer support assistant
specializing in company policies.

Answer the customer's policy question using ONLY the
retrieved policy context.

Follow these rules:

1. Identify exactly which policy information the customer
   is asking about.

2. Use only facts explicitly provided in the retrieved
   policy context.

3. Do not use outside knowledge.

4. Do not invent policy conditions, exceptions, timelines,
   refund amounts, delivery times, eligibility rules, or
   other information.

5. Include important conditions, limitations, and exceptions
   when they affect the answer.

6. You may perform simple logical reasoning when the conclusion
   is directly supported by the policy context.

7. If multiple policy documents are retrieved, use only the
   documents relevant to the customer's specific question.

8. Do not substitute information from an unrelated policy.

9. If the policy context does not contain enough information
   to answer the customer's specific question, respond:

   "I don't have enough information in the available policy data."

Give a direct, concise, customer-friendly answer.
""".strip()
            ),
            (
                "human",
                """
Customer question:
{question}

Retrieved policy context:
{context}

Answer the customer's question using only the retrieved policy context.
""".strip()
            )
        ]
    )

    llm = get_llm()

    chain = prompt | llm

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

    active_product_names = state.get(
    "active_product_names",
    []
    )

    active_products = "\n".join(
    f"- {product_name}"
    for product_name in active_product_names
    )

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

9. RECENCY RULE:

Use the MOST RECENTLY ACTIVE PRODUCTS only when the latest
question depends on a reference to previously discussed products,
such as "it", "that one", "those", "these", "them", "which one",
"first", "second", or "third".

Do NOT use the active products when the latest question is already
a standalone request that clearly identifies a new product category,
product type, policy topic, or product name.

For example:

Active products:
FitPulse Smartwatch 154
FitPulse Smartwatch 174
FitPulse Smartwatch 194

Question:
Which laptop should I buy?

Output:
laptop recommendations

The new question explicitly asks about laptops, so the previous
smartwatches must not be used.

If the question contains an ambiguous reference such as
"Which of those is cheapest?", use the active products first.

Only use older conversation history when the active products
cannot resolve the reference.

Return ONLY the rewritten search query.
""".strip()
            ),
            (
                "human",
                """
Conversation history:
{history}

Most recently active products:
{active_products}

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
            "active_products": active_products,
            "question": question
        }
    )

    retrieval_query = response.content.strip()

    print(f"ORIGINAL QUERY: {question}")
    print(f"RETRIEVAL QUERY: {retrieval_query}")

    return {
        "retrieval_query": retrieval_query
    }


if __name__ == "__main__":

    state = {
        "retrieval_query": "NovaTech Laptop 151",
    }

    result = product_agent_node(state)

    print("\nDOCUMENTS:")

    for document in result["documents"]:
        print(
            document.metadata.get("product_name")
        )

    print(
        "\nACTIVE PRODUCTS:",
        result["active_product_names"]
    )