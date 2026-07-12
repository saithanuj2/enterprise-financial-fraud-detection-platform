from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CASE_DIR = Path("rag_assistant/data/case_summaries")
VECTOR_DIR = Path("rag_assistant/vector_store")

VECTOR_DIR.mkdir(parents=True, exist_ok=True)

documents = []

print("Reading fraud case summaries...")

for file_path in CASE_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    documents.append(
        Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
    )

print(f"Loaded {len(documents)} documents.")

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(documents, embeddings)

vector_store.save_local(str(VECTOR_DIR))

print(f"Vector store saved successfully at: {VECTOR_DIR}")