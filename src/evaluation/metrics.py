import re


def normalize_text(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(expected_answer, generated_answer):
    """
    Calculate token-level F1 between expected and generated answers.
    This allows reasonable paraphrasing instead of requiring
    an exact string match.
    """

    expected_tokens = set(
        normalize_text(expected_answer).split()
    )

    generated_tokens = set(
        normalize_text(generated_answer).split()
    )

    if not expected_tokens or not generated_tokens:
        return 0.0

    common_tokens = expected_tokens.intersection(
        generated_tokens
    )

    if not common_tokens:
        return 0.0

    precision = (
        len(common_tokens)
        / len(generated_tokens)
    )

    recall = (
        len(common_tokens)
        / len(expected_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def answer_correct(expected_answer, generated_answer):
    """
    Determine whether the generated answer is sufficiently
    similar to the expected answer.
    """

    score = token_f1(
        expected_answer,
        generated_answer
    )

    return score >= 0.70


def retrieval_relevance(documents, relevant_source):
    """
    Check whether the expected source document
    was retrieved.
    """

    if relevant_source is None:
        return None

    for document in documents:

        source = document.metadata.get(
            "source",
            ""
        )

        if source.endswith(relevant_source):
            return True

    return False


def abstention_correct(generated_answer, relevant_source):
    """
    Check whether the model correctly refuses to answer
    an unanswerable question.
    """

    if relevant_source is not None:
        return None

    answer = normalize_text(
        generated_answer
    )

    refusal_phrases = [
        "i dont have enough information",
        "not provide information",
        "cannot be found",
        "not available in the provided documents",
        "do not provide information",
        "does not provide information",
        "no information"
    ]

    return any(
        phrase in answer
        for phrase in refusal_phrases
    )