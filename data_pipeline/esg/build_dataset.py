"""Build logical-structure RAG chunks from raw KOTRA ethics PDFs."""

from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

from chunk_articles import chunk_document
from config import DOCUMENT_CONFIG
from parse_pdf import concatenate_pages, extract_pages
from utils import (
    find_pdf_for_document,
    normalize_text,
    safe_filename_stem,
    save_json,
    save_jsonl,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ORIGIN_PDF_LABELS = {
    "공직자의 이해충돌 방지법": "이해충돌방지법",
    "부정청탁 및 금품등 수수의 금지에 관한 법률": "청탁금지법",
    "직장 내 인권침해 예방지침": "직장 내 인권침해 예방지침",
    "KOTRA 임직원 이해충돌 방지제도 운영지침": "KOTRA 임직원 이해충돌 방지제도 운영지침",
    "KOTRA 임직원 행동강령": "KOTRA 임직원 행동강령",
}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    chunks_by_document: dict[str, list[dict]] = {}

    for document_name, config in tqdm(DOCUMENT_CONFIG.items(), desc="Chunk PDFs"):
        pdf_path = find_pdf_for_document(RAW_DIR, document_name, config["source_keywords"])
        pages = extract_pages(pdf_path)
        full_text, page_spans = concatenate_pages(pages)
        chunks = chunk_document(document_name, pdf_path, full_text, page_spans, config)
        chunks_by_document[document_name] = chunks
        all_chunks.extend(chunks)

    save_jsonl(PROCESSED_DIR / "chunks.jsonl", all_chunks)
    save_json(PROCESSED_DIR / "rag_documents.json", [to_langchain_document_record(chunk) for chunk in all_chunks])
    write_document_json_files(PROCESSED_DIR / "by_document", chunks_by_document)
    write_summary(PROCESSED_DIR / "chunks_summary.csv", all_chunks)
    write_preview(PROCESSED_DIR / "chunks_preview.md", all_chunks)

    print(f"Wrote {len(all_chunks)} chunks to {PROCESSED_DIR}")


def write_document_json_files(output_dir: Path, chunks_by_document: dict[str, list[dict]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.json"):
        stale_file.unlink()

    for document_name, chunks in chunks_by_document.items():
        output_path = output_dir / f"{safe_filename_stem(document_name)}.json"
        save_json(output_path, [to_langchain_document_record(chunk) for chunk in chunks])


def to_langchain_document_record(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    logical_page_num = get_logical_page_num(chunk)
    origin_pdf = ORIGIN_PDF_LABELS.get(chunk["document_name"], chunk["document_name"])
    record_metadata = {
        "source": chunk["source_file"],
        "origin_pdf": origin_pdf,
        "page_num": logical_page_num,
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "chunk_id": chunk["chunk_id"],
        "document_name": chunk["document_name"],
        "section_type": chunk["section_type"],
        "title": chunk["title"],
        "chapter": chunk["chapter"],
        "chapter_number": chunk["chapter_number"],
        "article": chunk["article"],
        "article_number": chunk["article_number"],
        "article_title": chunk["article_title"],
        "appendix": chunk["appendix"],
        "content_format": chunk["content_format"],
        "is_deleted_article": metadata.get("is_deleted_article", False),
        "is_appendix": metadata.get("is_appendix", False),
        "is_table": metadata.get("is_table", False),
        "structure_key": metadata.get("structure_key"),
    }
    return {
        "page_content": chunk["content"],
        "metadata": pinecone_safe_metadata(record_metadata),
    }


def pinecone_safe_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != "" and is_pinecone_metadata_value(value)
    }


def is_pinecone_metadata_value(value: object) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    return False


def get_logical_page_num(chunk: dict) -> str:
    return chunk.get("title")


def write_summary(path: Path, chunks: list[dict]) -> None:
    columns = [
        "chunk_id",
        "document_name",
        "section_type",
        "chapter",
        "article",
        "appendix",
        "title",
        "content_length",
        "content_format",
        "page_start",
        "page_end",
        "is_deleted_article",
        "is_appendix",
        "is_table",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for chunk in chunks:
            metadata = chunk["metadata"]
            writer.writerow(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_name": chunk["document_name"],
                    "section_type": chunk["section_type"],
                    "chapter": chunk["chapter"],
                    "article": chunk["article"],
                    "appendix": chunk["appendix"],
                    "title": chunk["title"],
                    "content_length": len(chunk["content"]),
                    "content_format": chunk["content_format"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "is_deleted_article": metadata["is_deleted_article"],
                    "is_appendix": metadata["is_appendix"],
                    "is_table": metadata["is_table"],
                }
            )


def write_preview(path: Path, chunks: list[dict]) -> None:
    lines: list[str] = []
    current_doc: str | None = None
    for chunk in chunks:
        if chunk["document_name"] != current_doc:
            if lines:
                lines.append("")
            current_doc = chunk["document_name"]
            lines.append(f"# {current_doc}")
            lines.append("")
        lines.append(f"## {chunk['title']}")
        lines.append("")
        lines.append(chunk["content"].strip()[:4000])
        lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
