from src.loaders.json_loader import load_products as load_json_products
from src.loaders.csv_loader import load_products as load_csv_products
from src.loaders.markdown_loader import load_policy
from src.rag.chunker import split_policy


def load_all_documents():

    # Load JSON products
    json_documents = load_json_products()

    # Load CSV products
    csv_documents = load_csv_products()

    # Load policy
    policy_documents = load_policy()

    # Split policy into smaller sections
    policy_chunks = split_policy(policy_documents[0])

    # Combine everything
    all_documents = (
        json_documents
        + csv_documents
        + policy_chunks
    )

    return all_documents


if __name__ == "__main__":

    documents = load_all_documents()

    print(f"Total documents: {len(documents)}")

    print("\nFirst document:")
    print(documents[0].page_content)

    print("\nMetadata:")
    print(documents[0].metadata)