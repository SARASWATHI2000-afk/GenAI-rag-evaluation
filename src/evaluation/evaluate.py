import json
import os
import pandas as pd

from src.generation.generator import generate_answer

from src.evaluation.metrics import (
    answer_correct,
    retrieval_relevance,
    abstention_correct
)

from src.evaluation.faithfulness import (
    check_faithfulness
)


DATASET_PATH = "data/evaluation/evaluation_dataset.json"


def load_evaluation_dataset():
    """Load evaluation questions from the JSON dataset."""

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_rag():
    """
    Run the complete RAG evaluation pipeline.

    Metrics:
    - Answer correctness
    - Retrieval relevance
    - Abstention accuracy
    - Faithfulness
    """

    dataset = load_evaluation_dataset()

    results = []

    for item in dataset:

        question = item["question"]
        expected_answer = item["expected_answer"]
        relevant_source = item["relevant_source"]

        print("=" * 70)
        print(
            f"Question {item['id']}: {question}"
        )

        # --------------------------------------------------
        # Run RAG pipeline
        # --------------------------------------------------

        answer, documents = generate_answer(
            question,
            top_k=3,
            top_n=2
        )

        # --------------------------------------------------
        # Retrieval evaluation
        # --------------------------------------------------

        retrieval_result = retrieval_relevance(
            documents,
            relevant_source
        )

        # --------------------------------------------------
        # Answer correctness
        # --------------------------------------------------

        if relevant_source is None:

            # Question is intentionally unanswerable.
            answer_result = abstention_correct(
                answer,
                relevant_source
            )

        else:

            answer_result = answer_correct(
                expected_answer,
                answer
            )

        # --------------------------------------------------
        # Abstention evaluation
        # --------------------------------------------------

        abstention_result = abstention_correct(
            answer,
            relevant_source
        )

        # --------------------------------------------------
        # Faithfulness evaluation
        # --------------------------------------------------

        faithfulness_result = check_faithfulness(
            question,
            answer,
            documents
        )

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        result = {
            "id": item["id"],
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "answer_correct": answer_result,
            "retrieval_relevant": retrieval_result,
            "abstention_correct": abstention_result,
            "faithful": faithfulness_result,
            "sources": [
                document.metadata.get("source")
                for document in documents
            ]
        }

        results.append(result)

        # --------------------------------------------------
        # Print individual evaluation
        # --------------------------------------------------

        print("\nExpected answer:")
        print(expected_answer)

        print("\nGenerated answer:")
        print(answer)

        print(
            f"\nAnswer correct: {answer_result}"
        )

        print(
            f"Retrieval relevant: "
            f"{retrieval_result}"
        )

        print(
            f"Faithfulness: "
            f"{faithfulness_result}"
        )

        print(
            f"Abstention correct: "
            f"{abstention_result}"
        )

    return results


def calculate_metrics(results):
    """Calculate aggregate evaluation metrics."""

    # --------------------------------------------------
    # Answer correctness
    # --------------------------------------------------

    answer_results = [
        result["answer_correct"]
        for result in results
    ]

    answer_accuracy = (
        sum(answer_results)
        / len(answer_results)
        if answer_results
        else 0
    )

    # --------------------------------------------------
    # Retrieval relevance
    # --------------------------------------------------

    retrieval_results = [
        result["retrieval_relevant"]
        for result in results
        if result["retrieval_relevant"] is not None
    ]

    retrieval_accuracy = (
        sum(retrieval_results)
        / len(retrieval_results)
        if retrieval_results
        else 0
    )

    # --------------------------------------------------
    # Abstention accuracy
    # --------------------------------------------------

    abstention_results = [
        result["abstention_correct"]
        for result in results
        if result["abstention_correct"] is not None
    ]

    abstention_accuracy = (
        sum(abstention_results)
        / len(abstention_results)
        if abstention_results
        else 0
    )

    # --------------------------------------------------
    # Faithfulness
    # --------------------------------------------------

    faithfulness_results = [
        result["faithful"]
        for result in results
    ]

    faithfulness_accuracy = (
        sum(faithfulness_results)
        / len(faithfulness_results)
        if faithfulness_results
        else 0
    )

    return {
        "answer_accuracy": answer_accuracy,
        "retrieval_accuracy": retrieval_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "faithfulness_accuracy": faithfulness_accuracy
    }
def save_results(results, metrics):
    """Save evaluation results to an Excel file."""

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "evaluation_results.xlsx"
    )

    # Detailed question-level results
    detailed_results = []

    for result in results:
        detailed_results.append({
            "Question ID": result["id"],
            "Question": result["question"],
            "Expected Answer": result["expected_answer"],
            "Generated Answer": result["generated_answer"],
            "Answer Correct": result["answer_correct"],
            "Retrieval Relevant": result["retrieval_relevant"],
            "Abstention Correct": result["abstention_correct"],
            "Faithful": result["faithful"],
            "Sources": ", ".join(
                str(source)
                for source in result["sources"]
                if source
            )
        })

    detailed_df = pd.DataFrame(detailed_results)

    # Overall summary
    summary_data = {
        "Metric": [
            "Total Questions",
            "Answer Correctness",
            "Retrieval Relevance",
            "Abstention Accuracy",
            "Faithfulness"
        ],
        "Score": [
            len(results),
            f"{metrics['answer_accuracy']:.2%}",
            f"{metrics['retrieval_accuracy']:.2%}",
            f"{metrics['abstention_accuracy']:.2%}",
            f"{metrics['faithfulness_accuracy']:.2%}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # Write both sheets to Excel
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        detailed_df.to_excel(
            writer,
            sheet_name="Detailed Results",
            index=False
        )

    print(
        f"\nEvaluation results saved to: {output_path}"
    )

if __name__ == "__main__":

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    results = evaluate_rag()

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    metrics = calculate_metrics(results)

    # Save results to Excel
    save_results(results, metrics)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total questions: "
        f"{len(results)}"
    )

    print(
        f"Answer correctness: "
        f"{metrics['answer_accuracy']:.2%}"
    )

    print(
        f"Retrieval relevance: "
        f"{metrics['retrieval_accuracy']:.2%}"
    )

    print(
        f"Abstention accuracy: "
        f"{metrics['abstention_accuracy']:.2%}"
    )

    print(
        f"Faithfulness: "
        f"{metrics['faithfulness_accuracy']:.2%}"
    )

    print("=" * 70)