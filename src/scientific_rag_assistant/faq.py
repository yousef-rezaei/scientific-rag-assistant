import json
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

FAQ_FILE = BASE_DIR / "data" / "faqs.json"

COLLECTION_NAME = "scientific_faq"


# --------------------------------------------------
# 2. Embedding model
# --------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


@lru_cache(maxsize=1)
def get_embeddings():
    """Load and cache the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# --------------------------------------------------
# 3. PostgreSQL connection
# --------------------------------------------------

def get_database_url():
    """Load PostgreSQL connection URL from .env."""

    load_dotenv(BASE_DIR / ".env")

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please configure the .env file."
        )

    return database_url


def get_vector_store():
    """Connect to PostgreSQL + pgvector."""

    database_url = get_database_url()

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=database_url,
        use_jsonb=True,
        create_extension=False,
    )


# --------------------------------------------------
# 4. Index FAQ
# --------------------------------------------------

def index_faq():
    """Read FAQ records and store their embeddings."""

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError(
            "FAQ data must be a non-empty list."
        )

    documents = []
    ids = []
    seen_ids = set()

    for faq in faqs:

        if not isinstance(faq, dict):
            raise ValueError(
                "Every FAQ record must be an object."
            )

        faq_id = str(faq["id"]).strip()

        question = faq["question"]
        answer = faq["answer"]

        if not isinstance(question, str):
            raise ValueError("FAQ question must be text.")

        if not isinstance(answer, str):
            raise ValueError("FAQ answer must be text.")

        question = question.strip()
        answer = answer.strip()

        if not faq_id or not question or not answer:
            raise ValueError(
                "FAQ fields cannot be empty."
            )

        if faq_id in seen_ids:
            raise ValueError(
                f"Duplicate FAQ ID: {faq_id}"
            )

        seen_ids.add(faq_id)

        # Stable ID for PostgreSQL upsert
        document_id = f"faq:{faq_id}"

        ids.append(document_id)

        document = Document(
            page_content=question,
            metadata={
                "faq_id": faq_id,
                "answer": answer,
                "source": "data/faqs.json",
            },
        )

        documents.append(document)

    # Connect to PostgreSQL
    vector_store = get_vector_store()

    # Generate embeddings and insert/update records
    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    print(
        f"Indexed {len(documents)} FAQ records "
        "into PostgreSQL."
    )


# --------------------------------------------------
# 5. Semantic FAQ search
# --------------------------------------------------

def search_faq(question: str):
    """Find the nearest FAQ questions in PostgreSQL."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question.strip(),
        k=3,
    )

    if not results:
        print(
            "No FAQ records found. "
            "Run the index command first."
        )
        return

    print("\nTop FAQ matches:\n")

    for rank, (document, distance) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank}. {document.page_content}"
        )

        print(
            f"   Cosine Distance: {distance:.4f}"
        )

        print(
            f"   FAQ ID: {document.metadata['faq_id']}"
        )

        print()

    best_document = results[0][0]

    print(
        "Suggested answer "
        "(relevance not yet validated):"
    )

    print(
        best_document.metadata["answer"]
    )