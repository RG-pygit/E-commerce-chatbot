from pathlib import Path
from langchain_core.documents import Document

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "company_policies.md"


def load_policy():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        content = file.read()

    document = Document(
        page_content=content,
        metadata={
            "source": "company_policies.md",
            "source_type": "policy"
        }
    )

    return [document]


if __name__ == "__main__":
    documents = load_policy()

    print(f"Total documents: {len(documents)}")

    print("\nCONTENT:")
    print(documents[0].page_content)

    print("\nMETADATA:")
    print(documents[0].metadata)