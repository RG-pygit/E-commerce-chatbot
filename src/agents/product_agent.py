from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.rag.generator import get_llm

from src.agents.product_tools import (
    semantic_product_search,
    exact_product_lookup,
    multiple_product_lookup
)
class ProductAgentDecision(BaseModel):

    tool: Literal[
        "semantic_search",
        "exact_lookup",
        "multiple_lookup"
    ]

    query: str | None = None

    product_names: list[str] = Field(
        default_factory=list
    )

def product_agent_decide(
    question: str
) -> ProductAgentDecision:

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        ProductAgentDecision
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Product Agent for an e-commerce chatbot.

Your job is to decide which product retrieval capability
should handle the customer's request.

Available capabilities:

semantic_search:
Use for broad product discovery, recommendations, or searches
where no exact product name is known.

Examples:
- Recommend a laptop
- Show me some smartwatches
- I need a laptop with good battery life


exact_lookup:
Use when exactly ONE complete product name is provided.

Examples:
- Tell me about NovaTech Laptop 151
- What is the price of FitPulse Smartwatch 154?


multiple_lookup:
Use when TWO OR MORE complete product names are provided.

Examples:
- Compare NovaTech Laptop 171 and NovaTech Laptop 151
- Compare FitPulse Smartwatch 154 and FitPulse Smartwatch 174


Important:

Do not invent product names.

For semantic_search:
- Put the useful search text in query.
- Leave product_names empty.

For exact_lookup:
- Put the exact product name in product_names.
- query may be empty.

For multiple_lookup:
- Put all exact product names in product_names.
- query may be empty.
""".strip()
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    chain = prompt | structured_llm

    return chain.invoke(
        {
            "question": question
        }
    )

def run_product_agent(question: str):
    """
    Decide which product tool to use,
    execute that tool,
    and return the retrieved documents.
    """

    decision = product_agent_decide(
        question
    )

    print(
        f"PRODUCT AGENT TOOL: {decision.tool}"
    )

    if decision.tool == "semantic_search":

        documents = semantic_product_search(
            decision.query
        )

    elif decision.tool == "exact_lookup":

        if not decision.product_names:
            return []

        documents = exact_product_lookup(
            decision.product_names[0]
        )

    elif decision.tool == "multiple_lookup":

        if not decision.product_names:
            return []

        documents = multiple_product_lookup(
            decision.product_names
        )

    else:

        documents = []

    return documents

##############################################
if __name__ == "__main__":

    test_questions = [
        "Recommend a laptop",
        "Tell me about NovaTech Laptop 151",
        (
            "Compare NovaTech Laptop 171 "
            "and NovaTech Laptop 151"
        )
    ]

    for question in test_questions:

        print("\n" + "=" * 60)
        print("QUESTION")
        print(question)

        documents = run_product_agent(
            question
        )

        print("\nDOCUMENTS RETURNED")

        for document in documents:

            print(
                "-",
                document.metadata.get(
                    "product_name"
                )
            )