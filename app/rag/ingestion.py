from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


def load_knowledge_chunks(
    directory: Path,
) -> list[Document]:

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory not found: {directory}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )

    chunks = []

    for path in sorted(directory.glob("*.md")):
        if path.stat().st_size > 2_000_000:
            raise ValueError(
                f"Knowledge file exceeds size limit: {path.name}"
            )

        content = path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not content:
            continue

        parts = splitter.split_text(content)

        for index, part in enumerate(parts, start=1):
            chunks.append(
                Document(
                    page_content=part,
                    metadata={
                        "source": path.name,
                        "chunk_id": (
                            f"{path.stem}:{index:03d}"
                        ),
                    },
                )
            )

    if not chunks:
        raise ValueError("No usable knowledge chunks")

    return chunks
