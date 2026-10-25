from typing import TypedDict
from langchain_core.documents import Document


class ChatbotState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """

    question: str
    route: str
    documents: list[Document]
    answer: str


