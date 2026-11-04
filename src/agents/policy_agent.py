from src.agents.policy_tools import (
    semantic_policy_search
)


def run_policy_agent(query: str):
    """
    Execute the appropriate policy capability.

    Currently the Policy Agent has one tool:
    semantic policy search.
    """

    print(
        "POLICY AGENT TOOL: semantic_policy_search"
    )

    documents = semantic_policy_search(
        query
    )

    return documents

if __name__ == "__main__":

    test_queries = [
        "delivery time",
        "refund processing",
        "cancel order after shipping",
        "return product after 20 days"
    ]

    for query in test_queries:

        print("\n" + "=" * 60)
        print("QUERY")
        print(query)

        documents = run_policy_agent(
            query
        )

        print("\nDOCUMENTS RETURNED")

        for document in documents:

            print(
                "-",
                document.metadata.get(
                    "policy_type"
                )
            )