from langchain_text_splitters import MarkdownHeaderTextSplitter


def split_policy(document):

    headers_to_split_on = [
        ("##", "policy_type")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    chunks = splitter.split_text(document.page_content)

    # Preserve metadata from original document
    for chunk in chunks:
        chunk.metadata.update(document.metadata)

    return chunks

if __name__ == "__main__":

    from src.loaders.markdown_loader import load_policy

    documents = load_policy()

    policy_document = documents[0]

    chunks = split_policy(policy_document)

    print(f"Total policy chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n----------------")
        print("CONTENT:")
        print(chunk.page_content)

        print("\nMETADATA:")
        print(chunk.metadata)
