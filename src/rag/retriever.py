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


if __name__ == "__main__":

    query="Can I return a product after 20 days?"

    results = search(query)

    print(f"\nQuestion: {query}")

    print("\nRetrieved documents:")

    for i, document in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print(f"RESULT {i}")
        print("=" * 60)

        print("\nCONTENT:")
        print(document.page_content)

        print("\nMETADATA:")
        print(document.metadata)