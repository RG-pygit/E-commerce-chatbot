from src.rag.retriever import (
    search_by_source_type,
    search_exact_product,
    search_exact_products,
)


def semantic_product_search(query: str, k: int = 3):
    """
    Search the product catalog semantically.

    Use this for broad product questions such as:
    - recommend a laptop
    - show me some smartwatches
    - laptop with good battery life
    """

    return search_by_source_type(
        query=query,
        source_type="product",
        k=k
    )


def exact_product_lookup(product_name: str):
    """
    Retrieve one exact product using its full product name.

    Example:
    NovaTech Laptop 151
    """

    return search_exact_product(
        product_name
    )


def multiple_product_lookup(product_names: list[str]):
    """
    Retrieve multiple exact products.

    Use this for comparison questions.

    Example:
    [
        "NovaTech Laptop 171",
        "NovaTech Laptop 181",
        "NovaTech Laptop 151"
    ]
    """

    return search_exact_products(
        product_names
    )
