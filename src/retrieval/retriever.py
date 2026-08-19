from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = "data/chroma_db"


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name="sustainability_documents",
        embedding_function=embeddings,
    )

    return vector_store


def retrieve_documents(query, k=3):
    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results


if __name__ == "__main__":
    query = "What are the company's sustainability goals?"

    results = retrieve_documents(query)

    print(f"Query: {query}")
    print(f"Retrieved documents: {len(results)}")

    for index, (document, score) in enumerate(results, start=1):
        print("-" * 60)
        print(f"Result {index}")
        print(f"Similarity score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Content: {document.page_content}")