"""
Integration tests for the Phase 1 e-commerce chatbot.

These tests invoke the actual LangGraph workflow, including:

- intent routing
- query rewriting
- Chroma retrieval
- conversation memory
- exact-product retrieval
- local LLM generation through Ollama
"""

from uuid import uuid4

from langchain_core.messages import HumanMessage

from src.graph.workflow import build_graph


def create_config():
    """
    Create a unique conversation thread for each test.

    This prevents memory from one test from affecting another test.
    """

    return {
        "configurable": {
            "thread_id": f"test-{uuid4()}"
        }
    }


def invoke_graph(graph, question, config):
    """
    Invoke the graph with one user message.
    """

    input_state = {
        "question": question,
        "messages": [
            HumanMessage(content=question)
        ]
    }

    return graph.invoke(
        input_state,
        config=config
    )


def get_source_types(result):
    """
    Return all source_type metadata values from retrieved documents.
    """

    return {
        document.metadata.get("source_type")
        for document in result.get("documents", [])
    }


def get_policy_types(result):
    """
    Return all policy_type metadata values from retrieved documents.
    """

    return {
        document.metadata.get("policy_type")
        for document in result.get("documents", [])
        if document.metadata.get("policy_type")
    }


def get_product_names(result):
    """
    Return all product_name metadata values from retrieved documents.
    """

    return [
        document.metadata.get("product_name")
        for document in result.get("documents", [])
        if document.metadata.get("product_name")
    ]

def get_numeric_field(document, field_name):
    """
    Extract an integer field from a Document's page_content.

    Example:
    field_name = "price_inr"
    line = "price_inr: 91399"
    returns 91399
    """

    prefix = f"{field_name}:"

    for line in document.page_content.splitlines():

        if line.startswith(prefix):

            value = line.split(":", 1)[1].strip()

            # Handles values such as:
            # "91399"
            # "32 GB"
            # "Up to 12 hours"
            digits = "".join(
                character
                for character in value
                if character.isdigit()
            )

            if digits:
                return int(digits)

    return None

# ------------------------------------------------------------------
# SINGLE-TURN TESTS
# ------------------------------------------------------------------


def test_greeting_routes_to_conversation():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question="Hi",
        config=config
    )

    assert result["route"] == "conversation"

    assert result["answer"]

    # Greeting route should not retrieve documents
    assert result.get("documents", []) == []


def test_mixed_greeting_product_query_routes_to_rag():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question=(
            "Hi, can you recommend a laptop "
            "with good battery life?"
        ),
        config=config
    )

    assert result["route"] == "rag"

    assert "product" in get_source_types(result)

    assert len(result.get("documents", [])) > 0


def test_product_query_retrieves_product_documents():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question="Which laptop should I buy?",
        config=config
    )

    assert result["route"] == "rag"

    assert "product" in get_source_types(result)

    product_names = get_product_names(result)

    assert len(product_names) > 0

    assert all(
        "Laptop" in product_name
        for product_name in product_names
    )


def test_delivery_policy_is_retrieved():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question="How long does delivery take?",
        config=config
    )

    assert result["route"] == "rag"

    assert "policy" in get_source_types(result)

    assert "Delivery Policy" in get_policy_types(result)


def test_cancellation_policy_is_retrieved():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question=(
            "Can I cancel an order after it has shipped?"
        ),
        config=config
    )

    assert result["route"] == "rag"

    assert "Cancellation Policy" in get_policy_types(result)


def test_return_policy_is_retrieved():
    graph = build_graph()
    config = create_config()

    result = invoke_graph(
        graph=graph,
        question=(
            "Can I return an eligible product after 20 days?"
        ),
        config=config
    )

    assert result["route"] == "rag"

    assert "Return Policy" in get_policy_types(result)


# ------------------------------------------------------------------
# MULTI-TURN MEMORY TESTS
# ------------------------------------------------------------------


def test_messages_are_saved_between_turns():
    graph = build_graph()
    config = create_config()

    first_result = invoke_graph(
        graph=graph,
        question="Hi",
        config=config
    )

    assert len(first_result["messages"]) == 2

    second_result = invoke_graph(
        graph=graph,
        question="Thank you for your help",
        config=config
    )

    # Two turns:
    # Human + AI + Human + AI
    assert len(second_result["messages"]) == 4

    message_types = [
        type(message).__name__
        for message in second_result["messages"]
    ]

    assert message_types == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage"
    ]


def test_smartwatch_follow_up_uses_exact_product():
    graph = build_graph()
    config = create_config()

    first_result = invoke_graph(
        graph=graph,
        question="Recommend a smartwatch.",
        config=config
    )

    assert first_result["route"] == "rag"
    assert "product" in get_source_types(first_result)

    second_result = invoke_graph(
        graph=graph,
        question=(
            "Tell me more about the smartwatch "
            "you recommended."
        ),
        config=config
    )

    assert second_result["route"] == "rag"

    product_names = get_product_names(second_result)

    # Exact-product retrieval should return one product
    assert len(product_names) == 1

    exact_product = product_names[0]

    assert "Smartwatch" in exact_product

    # The rewritten query should identify the same product
    assert exact_product in second_result["retrieval_query"]


def test_smartwatch_price_follow_up_uses_same_product():
    graph = build_graph()
    config = create_config()

    invoke_graph(
        graph=graph,
        question="Recommend a smartwatch.",
        config=config
    )

    details_result = invoke_graph(
        graph=graph,
        question=(
            "Tell me more about the smartwatch "
            "you recommended."
        ),
        config=config
    )

    recommended_products = get_product_names(details_result)

    assert len(recommended_products) == 1

    recommended_product = recommended_products[0]

    price_result = invoke_graph(
        graph=graph,
        question="How much does that smartwatch cost?",
        config=config
    )

    price_products = get_product_names(price_result)

    assert price_products == [recommended_product]

    assert recommended_product in price_result["retrieval_query"]


def test_laptop_position_follow_up_uses_exact_product():
    graph = build_graph()
    config = create_config()

    first_result = invoke_graph(
        graph=graph,
        question="Which laptop should I buy?",
        config=config
    )

    first_turn_products = get_product_names(first_result)

    assert len(first_turn_products) >= 3

    second_result = invoke_graph(
        graph=graph,
        question=(
            "How about the third laptop "
            "you recommended?"
        ),
        config=config
    )

    follow_up_products = get_product_names(second_result)

    # Exact retrieval should provide only one product document
    assert len(follow_up_products) == 1

    exact_product = follow_up_products[0]

    assert exact_product in second_result["retrieval_query"]



def test_compare_first_and_third_laptop_options():

    graph = build_graph()
    config = create_config()

    # Turn 1
    first_result = invoke_graph(
        graph=graph,
        question="Which laptop should I buy?",
        config=config
    )

    products = get_product_names(first_result)

    assert len(products) >= 3

    # Turn 2
    comparison_result = invoke_graph(
        graph=graph,
        question="Compare the first and third options.",
        config=config
    )

    assert comparison_result["route"] == "rag"
    assert comparison_result["answer"]

    print("Products retrieved in first turn:")
    print(products)

    print("Rewritten query:")
    print(comparison_result["retrieval_query"])

    print("Products retrieved for comparison:")
    print(get_product_names(comparison_result))

    print("Answer:")
    print(comparison_result["answer"])



def test_laptop_price_comparison():

    graph = build_graph()
    config = create_config()

    # Turn 1
    first_result = invoke_graph(
        graph=graph,
        question="Which laptop should I buy?",
        config=config
    )

    laptop_documents = first_result.get(
        "documents",
        []
    )

    assert len(laptop_documents) >= 2

    products_with_prices = []

    for document in laptop_documents:

        price = get_numeric_field(
            document,
            "price_inr"
        )

        product_name = document.metadata.get(
            "product_name"
        )

        if product_name and price is not None:
            products_with_prices.append(
                {
                    "product_name": product_name,
                    "price": price
                }
            )

    assert len(products_with_prices) >= 2

    expected_cheapest = min(
        products_with_prices,
        key=lambda product: product["price"]
    )

    # Turn 2
    comparison_result = invoke_graph(
        graph=graph,
        question="Which of those is the cheapest?",
        config=config
    )

    # Check products retrieved for the comparison question
    comparison_products = get_product_names(
    comparison_result
    )

    assert len(comparison_products) >= 2, (
    f"Expected at least 2 products for comparison, "
    f"but retrieved: {comparison_products}"
    )   

    answer = comparison_result["answer"].lower()

    assert (
        expected_cheapest["product_name"].lower()
        in answer
    ), (
        f"Expected cheapest product: "
        f"{expected_cheapest['product_name']} "
        f"({expected_cheapest['price']} INR). "
        f"Actual answer: {comparison_result['answer']}"
    )


def test_laptop_ram_comparison():

    graph = build_graph()
    config = create_config()

    first_result = invoke_graph(
        graph=graph,
        question="Which laptop should I buy?",
        config=config
    )

    documents = first_result.get(
        "documents",
        []
    )

    assert len(documents) >= 2

    products = []

    for document in documents:

        ram = get_numeric_field(
            document,
            "ram"
        )

        product_name = document.metadata.get(
            "product_name"
        )

        if product_name and ram is not None:
            products.append(
                {
                    "product_name": product_name,
                    "ram": ram
                }
            )

    assert len(products) >= 2

    maximum_ram = max(
        product["ram"]
        for product in products
    )

    # There could be a tie
    expected_products = {
        product["product_name"]
        for product in products
        if product["ram"] == maximum_ram
    }

    comparison_result = invoke_graph(
        graph=graph,
        question="Which of those has the most RAM?",
        config=config
    )

    comparison_products = get_product_names(
    comparison_result
)

    assert len(comparison_products) >= 2, (
    f"Expected multiple products for comparison, "
    f"but retrieved: {comparison_products}"
    )
    
    answer = comparison_result["answer"].lower()

    assert any(
        product.lower() in answer
        for product in expected_products
    ), (
        f"Expected one of: {expected_products}. "
        f"Maximum RAM: {maximum_ram} GB. "
        f"Actual answer: {comparison_result['answer']}"
    )



