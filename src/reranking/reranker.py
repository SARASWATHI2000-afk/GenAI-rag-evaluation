from sentence_transformers import CrossEncoder

from src.retrieval.retriever import retrieve_documents


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(query, documents):
    reranker = CrossEncoder(MODEL_NAME)

    pairs = [
        [query, document.page_content]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents


if __name__ == "__main__":
    query = "What are the company's sustainability goals?"

    retrieved_results = retrieve_documents(query, k=3)

    documents = [
        document
        for document, _ in retrieved_results
    ]

    reranked_results = rerank_documents(query, documents)

    print(f"Query: {query}")
    print(f"Retrieved documents: {len(documents)}")

    print("\nRERANKED RESULTS")

    for index, (document, score) in enumerate(
        reranked_results,
        start=1
    ):
        print("-" * 60)
        print(f"Rank: {index}")
        print(f"Reranker score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Content: {document.page_content}")