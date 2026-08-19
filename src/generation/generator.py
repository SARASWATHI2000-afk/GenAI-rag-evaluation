from langchain_ollama import ChatOllama

from src.retrieval.retriever import retrieve_documents
from src.reranking.reranker import rerank_documents


MODEL_NAME = "llama3.2:3b"


# Create the local Ollama LLM once at module level.
# This allows other evaluation components to reuse it.
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


def generate_answer(query, top_k=3, top_n=2):
    """
    Run the complete RAG pipeline:

    1. Retrieve documents
    2. Rerank documents
    3. Select top documents
    4. Build context
    5. Generate answer using Ollama
    """

    # --------------------------------------------------
    # Step 1: Retrieve candidate documents
    # --------------------------------------------------

    retrieved_results = retrieve_documents(
        query,
        k=top_k
    )

    documents = [
        document
        for document, _ in retrieved_results
    ]

    # --------------------------------------------------
    # Step 2: Rerank retrieved documents
    # --------------------------------------------------

    reranked_results = rerank_documents(
        query,
        documents
    )

    # --------------------------------------------------
    # Step 3: Select top documents
    # --------------------------------------------------

    selected_documents = [
        document
        for document, _
        in reranked_results[:top_n]
    ]

    # --------------------------------------------------
    # Step 4: Build context
    # --------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in selected_documents
    )

    # --------------------------------------------------
    # Step 5: Create grounded prompt
    # --------------------------------------------------

    prompt = f"""
You are a helpful assistant answering questions
about company sustainability.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context, say:

"I don't have enough information in the provided documents."

Do not make up facts.
Do not use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

    # --------------------------------------------------
    # Step 6: Generate answer using local Ollama
    # --------------------------------------------------

    response = llm.invoke(prompt)

    return response.content, selected_documents


if __name__ == "__main__":

    query = "What was the company's total revenue in 2025?"

    answer, documents = generate_answer(query)

    print("=" * 60)
    print("USER QUERY")
    print("=" * 60)
    print(query)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{index}. "
            f"{document.metadata.get('source')}"
        )