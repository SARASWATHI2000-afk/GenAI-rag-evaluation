from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import load_documents


def split_documents(documents):
    """Split documents into smaller chunks for retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print("-" * 60)
        print(f"Chunk {index}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Content: {chunk.page_content[:200]}...")