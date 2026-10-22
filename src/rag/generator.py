from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.rag.retriever import search


#MODEL_NAME = "llama3.2:3b"
MODEL_NAME = "gemma3:4b"


def get_llm():
    """
    Create the local Llama model through Ollama.
    """
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )


def format_context(documents):
    """
    Convert retrieved LangChain Documents
    into text that can be given to the LLM.
    """

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown")

        item_name = (
            document.metadata.get("product_name")
            or document.metadata.get("policy_type")
            or "Unknown"
        )

        context_parts.append(
            f"""
Context {i}
Source: {source}
Item: {item_name}

{document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question):
    """
    Complete RAG flow:

    Question
        -> retrieve documents
        -> build context
        -> send context + question to LLM
        -> generate answer
    """

    # STEP 1: Retrieve relevant documents from Chroma
    documents = search(question, k=3)

    # STEP 2: Convert retrieved documents into context
    context = format_context(documents)

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    print(context)

    # STEP 3: Create the prompt
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an e-commerce customer support assistant.

Answer the customer's question using ONLY the retrieved context.

Follow these rules:

1. Read the retrieved context carefully.

2. Identify the information in the context that is relevant
   to the customer's question.

3. Use facts explicitly provided in the retrieved context.

4. You may perform simple logical reasoning, comparisons,
   calculations, and conclusions when they are directly
   supported by the retrieved context.

5. When comparing values such as price, quantity, time,
   storage, RAM, battery life, ratings, or other numerical
   values, reason about them correctly before answering.

6. Do not invent product specifications, availability,
   prices, policies, conditions, or other information.

7. Do not use outside knowledge.

8. Include important conditions, limitations, or exceptions
   found in the retrieved context when they affect the answer.

9. If multiple retrieved items satisfy the customer's request,
   present the relevant options clearly.

10. If the retrieved context does not contain enough information
    to answer the question, respond:

    "I don't have enough information in the available product
    and policy data."

Give a direct, concise, customer-friendly answer.
""".strip()
        ),

        (
            "human",
            """
Customer question:
{question}

Retrieved context:
{context}

Answer the customer's question using only the retrieved context.
""".strip()
        )
    ]
)

    # STEP 4: Load local LLM
    llm = get_llm()

    # STEP 5: Connect prompt to LLM
    chain = prompt | llm

    # STEP 6: Generate answer
    response = chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    return response.content, documents


if __name__ == "__main__":

    question = input("\nAsk a product or policy question: ")

    answer, source_documents = generate_answer(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    for document in source_documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        item = (
            document.metadata.get("product_name")
            or document.metadata.get("policy_type")
            or "Unknown"
        )

        print(f"- {source}: {item}")