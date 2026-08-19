import re


def normalize_text(text):
    """
    Normalize text for comparison.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_faithfulness(question, answer, documents):
    """
    Check whether the generated answer is supported
    by the retrieved documents.

    Uses deterministic evidence matching to avoid
    instability from a small LLM-as-a-judge model.
    """

    # --------------------------------------------------
    # Build retrieved context
    # --------------------------------------------------

    context = " ".join(
        document.page_content
        for document in documents
    )

    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)

    # --------------------------------------------------
    # Handle correct abstention
    # --------------------------------------------------

    abstention_phrases = [
        "i dont have enough information",
        "not provided",
        "not available",
        "cannot be found",
        "does not provide information"
    ]

    if any(
        phrase in normalized_answer
        for phrase in abstention_phrases
    ):
        return True

    # --------------------------------------------------
    # Split answer into meaningful statements
    # --------------------------------------------------

    sentences = re.split(
        r"[.!?]",
        normalized_answer
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return False

    # --------------------------------------------------
    # Evidence matching
    # --------------------------------------------------

    supported_sentences = 0

    for sentence in sentences:

        words = sentence.split()

        # Ignore very small fragments
        if len(words) < 3:
            continue

        # Create n-grams from the answer
        ngrams = []

        for n in [5, 4, 3]:
            for i in range(len(words) - n + 1):
                ngram = " ".join(
                    words[i:i + n]
                )
                ngrams.append(ngram)

        # Check whether meaningful phrases occur
        # in the retrieved context
        matches = sum(
            1
            for ngram in ngrams
            if ngram in normalized_context
        )

        if matches >= 1:
            supported_sentences += 1

    # --------------------------------------------------
    # Final decision
    # --------------------------------------------------

    return supported_sentences == len(sentences)
