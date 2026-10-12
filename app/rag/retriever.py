from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:

    if not chunks:
        raise ValueError("Cannot index empty knowledge")

    store = InMemoryVectorStore(
        embedding=embeddings,
    )

    store.add_documents(chunks)

    return store


def retrieve_knowledge(
    store: InMemoryVectorStore,
    query: str,
    k: int = 3,
) -> list[dict]:

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not 1 <= k <= 10:
        raise ValueError("Invalid retrieval limit")

    documents = store.similarity_search(
        query,
        k=k,
    )

    return [
        {
            "source": doc.metadata["source"],
            "chunk_id": doc.metadata["chunk_id"],
            "content": doc.page_content,
        }
        for doc in documents
    ]
