from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.ingest import load_all_documents


# Store Chroma DB outside src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = PROJECT_ROOT / "vector_store"


def get_embedding_model():
    """
    Creates the embedding model used to convert
    document text into vectors.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store():
    """
    Loads all documents, creates embeddings,
    and stores them in Chroma.
    """

    print("Loading documents...")

    documents = load_all_documents()

    print(f"Documents loaded: {len(documents)}")
    print("Creating embeddings and vector store...")

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_PATH),
        collection_name="ecommerce_knowledge"
    )

    print("Vector store created successfully!")
    print(f"Saved at: {VECTOR_DB_PATH}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()