"""Shared helpers for PDF parsing and RAG chunk dataset generation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without destroying Korean legal markers."""
    text = unicodedata.normalize("NFC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_filename(value: str | Path) -> str:
    return unicodedata.normalize("NFC", Path(value).name)


def compact_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).lower()
    return re.sub(r"[\s_\-().·ㆍ,]+", "", normalized)


def find_pdf_for_document(raw_dir: Path, document_name: str, keywords: list[str]) -> Path:
    candidates = sorted(raw_dir.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    document_key = compact_key(document_name)
    best_score = -1
    best_path: Path | None = None
    for path in candidates:
        filename = normalize_filename(path)
        file_key = compact_key(filename)
        score = sum(1 for keyword in keywords if compact_key(keyword) in file_key)
        if document_key in file_key or file_key in document_key:
            score += 10
        if score > best_score:
            best_score = score
            best_path = path

    if best_path is None or best_score <= 0:
        available = ", ".join(normalize_filename(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not match PDF for {document_name}. Available PDFs: {available}"
        )
    return best_path


def sanitize_for_id(value: str) -> str:
    text = unicodedata.normalize("NFC", value).lower()
    replacements = {
        "kotra 임직원 행동강령": "kotra_code_of_conduct",
        "kotra 임직원 이해충돌 방지제도 운영지침": "kotra_conflict_interest_guideline",
        "공직자의 이해충돌 방지법": "public_official_conflict_interest_act",
        "부정청탁 및 금품등 수수의 금지에 관한 법률": "improper_solicitation_graft_act",
        "직장 내 인권침해 예방지침": "workplace_human_rights_guideline",
    }
    if text in replacements:
        return replacements[text]
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", text).strip("_")
    return text or "chunk"


def article_number_for_id(article_number: str | None) -> str:
    if not article_number:
        return "000"
    match = re.search(r"제\s*(\d+)\s*조(?:의\s*(\d+))?", article_number)
    if not match:
        return re.sub(r"\W+", "_", article_number)
    base = int(match.group(1))
    suffix = match.group(2)
    if suffix:
        return f"{base:03d}_{int(suffix)}"
    return f"{base:03d}"


def generate_chunk_id(
    document_name: str,
    section_type: str,
    article_number: str | None = None,
    appendix: str | None = None,
    index: int | None = None,
) -> str:
    prefix = sanitize_for_id(document_name)
    if article_number:
        return f"{prefix}_article_{article_number_for_id(article_number)}"
    if appendix:
        appendix_key = compact_key(appendix)
        appendix_key = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", appendix_key).strip("_")
        suffix = f"{index:03d}" if index is not None else appendix_key
        return f"{prefix}_{section_type}_{suffix}"
    suffix = f"{index:03d}" if index is not None else "000"
    return f"{prefix}_{section_type}_{suffix}"


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_filename_stem(value: str) -> str:
    text = unicodedata.normalize("NFC", value).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .") or "document"


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
