from pathlib import Path

from langchain_chroma import Chroma

from src.rag.vector_store import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = PROJECT_ROOT / "vector_store"


def load_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embedding_model,
        collection_name="ecommerce_knowledge"
    )

    return vector_store


def search(query, k=3):

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results

# this func is to search for exact product and not top 3 
def search_exact_product(product_name):

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        product_name,
        k=1,
        filter={
            "product_name": product_name
        }
    )

    return results

def find_exact_product_name(query):
    """
    Check whether the query contains a product name
    stored in the Chroma catalog metadata.

    Returns the exact stored product name when found.
    Otherwise returns None.
    """

    vector_store = load_vector_store()

    stored_data = vector_store.get(
        include=["metadatas"]
    )

    query_lower = query.strip().lower()

    product_names = set()

    for metadata in stored_data["metadatas"]:

        if not metadata:
            continue

        product_name = metadata.get("product_name")

        if product_name:
            product_names.add(product_name)

    # Check longer names first
    sorted_product_names = sorted(
        product_names,
        key=len,
        reverse=True
    )

    for product_name in sorted_product_names:

        if product_name.lower() in query_lower:
            return product_name

    return None

###### main  ######
if __name__ == "__main__":

    test_queries = [
        "Tell me about NovaTech Laptop 151",
        "What about FitPulse Smartwatch 194?",
        "Show details for AsterMobile Smartphone 182",
        "Recommend a smartwatch"
    ]

    for query in test_queries:

        product_name = find_exact_product_name(query)

        print("\nQuery:", query)
        print("Exact product:", product_name)


