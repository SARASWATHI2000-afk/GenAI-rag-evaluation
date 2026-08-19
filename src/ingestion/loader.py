from pathlib import Path

from langchain_community.document_loaders import TextLoader


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of our sample documents
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"


def load_documents():
    """Load all text documents from the sample_documents folder."""

    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        loaded_documents = loader.load()
        documents.extend(loaded_documents)

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    for document in documents:
        print("-" * 60)
        print(f"Source: {document.metadata.get('source')}")
        print(f"Characters: {len(document.page_content)}")