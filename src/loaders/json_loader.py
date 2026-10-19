import json
from pathlib import Path
from langchain_core.documents import Document

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "products_json.json"


def load_products():
    documents = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        products = json.load(file)

    for product in products:

        # Content that will eventually be embedded
        content = "\n".join(
            f"{key}: {value}"
            for key, value in product.items()
        )

        # Metadata describing this document
        metadata = {
            "source": "products.json",
            "source_type": "product",
            "product_id": product.get("product_id"),
            "product_name": product.get("name"),
            "category": product.get("category"),
            "brand": product.get("brand")
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents


if __name__ == "__main__":
    documents = load_products()

    print(f"Total documents: {len(documents)}")

    print("\nCONTENT:")
    print(documents[0].page_content)

    print("\nMETADATA:")
    print(documents[0].metadata)