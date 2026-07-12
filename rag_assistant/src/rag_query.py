from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

VECTOR_DIR = Path("rag_assistant/vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"


def load_vector_store() -> FAISS:
    """Load the local FAISS fraud-case vector store."""

    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_DIR}. "
            "Run build_vector_store.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_fraud_cases(
    vector_store: FAISS,
    question: str,
    k: int = 5,
):
    """Retrieve the fraud cases most relevant to the question."""

    return vector_store.similarity_search(question, k=k)


def build_context(documents) -> tuple[str, list[str]]:
    """Combine retrieved documents into context for the LLM."""

    context_parts = []
    sources = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown")
        sources.append(source)

        context_parts.append(
            f"""
Retrieved Fraud Case {index}
Source: {source}

{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts), sources


def generate_investigation_report(
    question: str,
    context: str,
    sources: list[str],
) -> str:
    """Generate a grounded investigator report with Ollama."""

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )

    source_list = "\n".join(f"- {source}" for source in sources)

    prompt = f"""
You are an enterprise financial fraud investigation assistant.

Answer the investigator's question using only the retrieved fraud cases below.

Rules:
- Do not invent transaction details.
- Do not claim retrieved examples prove fraud in a new transaction.
- Separate observed evidence from interpretation.
- If information is unavailable, say so.
- Keep the answer concise and professional.

Return the answer exactly in this format:

Investigation Summary:
<brief conclusion>

Main Risk Indicators:
- <indicator 1>
- <indicator 2>
- <indicator 3>

Similar Historical Patterns:
- <pattern 1>
- <pattern 2>

Recommended Investigator Actions:
- <action 1>
- <action 2>
- <action 3>

Retrieved Sources:
{source_list}

Investigator Question:
{question}

Retrieved Fraud Cases:
{context}
"""

    response = llm.invoke(prompt)
    return str(response.content)


def main() -> None:
    print("\nEnterprise Fraud Investigator RAG Assistant")
    print("Type 'exit' to close the application.\n")

    try:
        vector_store = load_vector_store()
    except Exception as error:
        print(f"Unable to load vector store: {error}")
        return

    while True:
        question = input("Ask a fraud investigation question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Closing Fraud Investigator Assistant.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            documents = retrieve_fraud_cases(
                vector_store=vector_store,
                question=question,
                k=5,
            )

            if not documents:
                print("No relevant fraud cases were found.\n")
                continue

            context, sources = build_context(documents)

            print("\nGenerating investigator report...\n")

            report = generate_investigation_report(
                question=question,
                context=context,
                sources=sources,
            )

            print("=" * 80)
            print(report)
            print("=" * 80)
            print()

        except Exception as error:
            print(f"\nRAG request failed: {error}\n")


if __name__ == "__main__":
    main()