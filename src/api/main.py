from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.graph.workflow import build_graph
from uuid import uuid4

app = FastAPI()

# Build the LangGraph once
graph = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.post("/chat")
def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    input_state = {
        "question": request.message,
        "messages": [
            HumanMessage(
                content=request.message
            )
        ]
    }

    result = graph.invoke(
        input_state,
        config=config
    )

    return {
        "answer": result["answer"],
        "route": result["route"],
        "thread_id": thread_id
    }