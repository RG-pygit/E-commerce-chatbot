from typing import TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatbotState(TypedDict):
    question: str
    retrieval_query: str
    route: str
    documents: list[Document]
    answer: str

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]