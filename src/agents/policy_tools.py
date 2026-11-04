from src.rag.retriever import search_by_source_type


def semantic_policy_search(query: str, k: int = 3):
    """
    Search only company policy documents.

    Examples:
    - How long does delivery take?
    - What is the refund policy?
    - Can I cancel after shipment?
    """

    return search_by_source_type(
        query=query,
        source_type="policy",
        k=k
    )

if __name__ == "__main__":

    results = semantic_policy_search(
        "delivery time"
    )

    for document in results:

        print(
            document.metadata.get(
                "policy_type"
            )
        )