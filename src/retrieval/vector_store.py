from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.loader import load_documents
from src.chunking.chunker import split_documents


PERSIST_DIRECTORY = "data/chroma_db"


def create_vector_store():
    # Load the original documents
    documents = load_documents()

    # Split documents into chunks
    chunks = split_documents(documents)

    # Create the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings and documents in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="sustainability_documents",
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print("Vector store created successfully.")
    print(f"ChromaDB location: {PERSIST_DIRECTORY}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()