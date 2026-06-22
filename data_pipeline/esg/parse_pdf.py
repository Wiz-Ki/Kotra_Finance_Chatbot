"""PDF text extraction helpers.

The chunker consumes one concatenated text stream, but page spans are retained so
each logical chunk can still report page_start/page_end for source tracing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from utils import normalize_text


def extract_pages(pdf_path: Path) -> list[dict]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required. Install dependencies with: pip install -r requirements.txt") from exc

    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            pages.append({"page": index, "text": normalize_text(text)})
    return remove_repeated_headers_footers(pages)


def remove_repeated_headers_footers(pages: list[dict], max_edge_lines: int = 2) -> list[dict]:
    """Remove exact repeated edge lines that appear on many pages."""
    if len(pages) < 4:
        return pages

    edge_counter: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        for line in lines[:max_edge_lines] + lines[-max_edge_lines:]:
            if 2 <= len(line) <= 80:
                edge_counter[line] += 1

    threshold = max(3, int(len(pages) * 0.45))
    repeated = {line for line, count in edge_counter.items() if count >= threshold}
    cleaned: list[dict] = []
    for page in pages:
        lines = page["text"].splitlines()
        kept = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            is_edge = i < max_edge_lines or i >= len(lines) - max_edge_lines
            if is_edge and stripped in repeated:
                continue
            kept.append(line)
        cleaned.append({"page": page["page"], "text": normalize_text("\n".join(kept))})
    return cleaned


def concatenate_pages(pages: Iterable[dict]) -> tuple[str, list[dict]]:
    chunks: list[str] = []
    page_spans: list[dict] = []
    cursor = 0
    for page in pages:
        if chunks:
            chunks.append("\n\n")
            cursor += 2
        text = page["text"]
        start = cursor
        chunks.append(text)
        cursor += len(text)
        page_spans.append({"page": page["page"], "start": start, "end": cursor})
    return "".join(chunks), page_spans


def pages_for_span(start: int, end: int, page_spans: list[dict]) -> tuple[int | None, int | None]:
    matches = [
        span["page"]
        for span in page_spans
        if span["start"] <= end and span["end"] >= start
    ]
    if not matches:
        return None, None
    return min(matches), max(matches)


def extract_tables_with_pdfplumber(pdf_path: Path) -> dict[int, list[list[list[str | None]]]]:
    """Optional table extraction hook for appendix-heavy PDFs."""
    try:
        import pdfplumber
    except ImportError:
        return {}

    tables_by_page: dict[int, list[list[list[str | None]]]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            if tables:
                tables_by_page[page_number] = tables
    return tables_by_page

