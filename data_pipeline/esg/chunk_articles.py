"""Logical Korean legal-document chunking for RAG ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import (
    APPENDIX_INLINE_PATTERN,
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    DELETED_ARTICLE_PATTERN,
)
from parse_pdf import pages_for_span
from utils import (
    first_non_empty_line,
    generate_chunk_id,
    normalize_filename,
    normalize_text,
)


@dataclass
class Section:
    title: str
    start: int
    end: int
    text: str


@dataclass
class Marker:
    title: str
    start: int


CHAPTER_RE = re.compile(CHAPTER_PATTERN)
ARTICLE_RE = re.compile(ARTICLE_PATTERN)
APPENDIX_RE = re.compile(APPENDIX_INLINE_PATTERN, re.MULTILINE)
DELETED_RE = re.compile(DELETED_ARTICLE_PATTERN)


def chunk_document(
    document_name: str,
    source_file: Path,
    text: str,
    page_spans: list[dict],
    config: dict,
) -> list[dict]:
    text = normalize_text(text)
    appendix_sections, body_text, body_offset = split_appendices(text, config)
    body_text = apply_drop_sections(body_text, config)

    chunks = chunk_articles(
        document_name=document_name,
        source_file=source_file,
        body_text=body_text,
        body_offset=body_offset,
        page_spans=page_spans,
        config=config,
    )

    if config.get("include_appendix"):
        start_index = len(chunks) + 1
        for index, section in enumerate(appendix_sections, start=start_index):
            chunks.append(
                build_appendix_chunk(
                    document_name=document_name,
                    source_file=source_file,
                    section=section,
                    page_spans=page_spans,
                    config=config,
                    index=index,
                )
            )
    return chunks


def split_appendices(text: str, config: dict) -> tuple[list[Section], str, int]:
    if not config.get("include_appendix"):
        return [], text, 0

    markers = find_tail_markers(text, config)
    if not markers:
        return [], text, 0

    body = text[: markers[0].start]
    sections: list[Section] = []
    for idx, marker in enumerate(markers):
        start = marker.start
        end = markers[idx + 1].start if idx + 1 < len(markers) else len(text)
        if is_requested_tail_section(marker.title, config):
            section_text = text[start:end]
            if marker.title.startswith("별표") and not section_text.lstrip().startswith("별표"):
                section_text = f"{marker.title}\n{section_text}"
            sections.append(Section(title=marker.title, start=start, end=end, text=section_text))
    return sections, body, 0


def find_tail_markers(text: str, config: dict) -> list[Marker]:
    markers: list[Marker] = []
    for match in APPENDIX_RE.finditer(text):
        title = normalize_text(match.group(1)).splitlines()[0].strip()
        markers.append(Marker(title=title, start=match.start()))

    for alias in config.get("appendix_aliases", []):
        pattern = re.compile(alias["pattern"], re.MULTILINE)
        for match in pattern.finditer(text):
            markers.append(Marker(title=alias["title"], start=match.start(1)))

    markers.sort(key=lambda marker: marker.start)
    deduped: list[Marker] = []
    for marker in markers:
        if deduped and marker.start == deduped[-1].start:
            if len(marker.title) > len(deduped[-1].title):
                deduped[-1] = marker
            continue
        deduped.append(marker)
    return deduped


def is_requested_appendix(title: str, config: dict) -> bool:
    return is_requested_tail_section(title, config) and not re.sub(r"\s+", "", title).startswith("부칙")


def is_requested_tail_section(title: str, config: dict) -> bool:
    keep_names = config.get("include_appendix_names")
    if keep_names:
        normalized = re.sub(r"\s+", "", title)
        return any(normalized.startswith(re.sub(r"\s+", "", name)) for name in keep_names)

    appendix_type = config.get("appendix_type", "")
    normalized = re.sub(r"\s+", "", title)
    if normalized.startswith("부칙") and not config.get("drop_sections"):
        return True
    if "별표" in appendix_type and normalized.startswith("별표"):
        return True
    if "별지" in appendix_type and normalized.startswith("별지"):
        return True
    return False


def apply_drop_sections(text: str, config: dict) -> str:
    cutoff = len(text)
    for marker in config.get("drop_sections", []):
        if re.sub(r"\s+", "", marker) == "부칙":
            marker_pattern = r"부\s*칙"
        elif marker.startswith("별지"):
            marker_pattern = r"<?별지\s*제?\s*\d*\s*호?"
        else:
            marker_pattern = re.escape(marker)
        pattern = re.compile(rf"(?:^|\n)\s*{marker_pattern}", re.MULTILINE)
        match = pattern.search(text)
        if match:
            cutoff = min(cutoff, match.start())
    return normalize_text(text[:cutoff])


def chunk_articles(
    document_name: str,
    source_file: Path,
    body_text: str,
    body_offset: int,
    page_spans: list[dict],
    config: dict,
) -> list[dict]:
    article_matches = valid_article_matches(body_text)
    if not article_matches:
        return []

    chapter_matches = (
        [match for match in CHAPTER_RE.finditer(body_text) if is_line_heading(body_text, match.start())]
        if config.get("has_chapter")
        else []
    )
    chunks: list[dict] = []

    for idx, match in enumerate(article_matches):
        next_start = article_matches[idx + 1].start() if idx + 1 < len(article_matches) else len(body_text)
        article_end = end_before_next_chapter(match.start(), next_start, chapter_matches)
        article_text = normalize_text(body_text[match.start():article_end])
        if not article_text:
            continue

        article = parse_article_heading(article_text)
        article_number, article_title = parse_article(article)
        chapter, chapter_number = current_chapter(match.start(), chapter_matches)
        is_deleted = bool(DELETED_RE.search(article))
        section_type = "deleted_article" if is_deleted else "article"
        article_text = normalize_article_text(article_text)

        content_parts = []
        if chapter:
            content_parts.append(chapter)
        content_parts.append(article_text)
        content = cleanup_spacing_artifacts(
            unwrap_soft_line_breaks(format_nested_letter_items(normalize_text("\n".join(content_parts))))
        )

        title = f"{chapter} {article}".strip() if chapter else article
        page_start, page_end = pages_for_span(
            body_offset + match.start(),
            body_offset + article_end,
            page_spans,
        )
        structure_parts = [document_name]
        if chapter:
            structure_parts.append(chapter)
        structure_parts.append(article)

        chunks.append(
            {
                "chunk_id": generate_chunk_id(document_name, section_type, article_number=article_number),
                "document_name": document_name,
                "source_file": normalize_filename(source_file),
                "section_type": section_type,
                "chapter": chapter,
                "chapter_number": chapter_number,
                "article": article,
                "article_number": article_number,
                "article_title": article_title,
                "appendix": None,
                "title": title,
                "content": content,
                "content_format": "text",
                "page_start": page_start,
                "page_end": page_end,
                "metadata": {
                    "document_type": "pdf",
                    "chunking_unit": "article",
                    "is_appendix": False,
                    "is_table": False,
                    "is_deleted_article": is_deleted,
                    "structure_key": " > ".join(structure_parts),
                },
            }
        )
    return chunks


def is_line_heading(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    return not text[line_start:start].strip()


def end_before_next_chapter(
    article_start: int,
    next_article_start: int,
    chapter_matches: list[re.Match],
) -> int:
    for chapter_match in chapter_matches:
        if article_start < chapter_match.start() < next_article_start:
            return chapter_match.start()
    return next_article_start


def format_nested_letter_items(text: str) -> str:
    lines = text.splitlines()
    formatted: list[str] = []
    in_each_mok_block = False
    in_letter_item = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append(line)
            continue

        is_heading = bool(re.match(r"^(제\s*\d+\s*장|제\s*\d+\s*조|#{1,6}\s)", stripped))
        is_number_item = bool(re.match(r"^\d+\.", stripped))
        is_circled_item = bool(re.match(r"^[①-⑳]", stripped))
        letter_item = re.match(r"^([가-하])\.\s+(.+)", stripped)
        has_each_mok = bool(re.search(r"각\s*목|각목", stripped))

        if is_heading:
            in_each_mok_block = False
            in_letter_item = False
            formatted.append(line)
            continue

        if is_number_item or is_circled_item:
            in_each_mok_block = has_each_mok
            in_letter_item = False
            formatted.append(line)
            continue

        if has_each_mok and not letter_item:
            in_each_mok_block = True
            formatted.append(line)
            continue

        if letter_item and in_each_mok_block:
            formatted.append(f"   {letter_item.group(1)}. {letter_item.group(2)}")
            in_letter_item = True
            continue

        if in_each_mok_block and in_letter_item:
            formatted.append(f"      {stripped}")
            continue

        formatted.append(line)

    return "\n".join(formatted)


def unwrap_soft_line_breaks(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    merged: list[str] = []
    in_each_mok_block = False
    in_letter_item = False
    for line in lines:
        raw_line = line.rstrip()
        stripped = raw_line.strip()

        starts_number_item = bool(re.match(r"^\d+\.", stripped) or re.match(r"^[①-⑳]", stripped))
        if starts_number_item:
            in_each_mok_block = bool(re.search(r"각\s*목|각목", stripped))
            in_letter_item = False

        if re.search(r"각\s*목|각목", stripped) and not re.match(r"^[가-하]\.\s+", stripped):
            in_each_mok_block = True

        letter_item = re.match(r"^([가-하])\.\s+(.+)", stripped)
        if in_each_mok_block and letter_item:
            merged.append(f"   {letter_item.group(1)}. {letter_item.group(2)}")
            in_letter_item = True
            continue

        if not merged or starts_new_logical_line(raw_line):
            merged.append(raw_line)
            continue

        previous = merged[-1].strip()
        if keeps_next_on_new_line(previous):
            merged.append(line.rstrip())
            continue

        merged[-1] = join_wrapped_line(merged[-1].rstrip(), stripped)

        if in_each_mok_block and in_letter_item:
            in_letter_item = True

    return "\n".join(merged)


def starts_new_logical_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^제\s*\d+\s*장\b", stripped)
        or re.match(r"^제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\([^)]+\)|<[^>]+>)", stripped)
        or re.match(r"^\d+\.", stripped)
        or re.match(r"^[①-⑳]", stripped)
        or re.match(r"^\s{3,}[가-하]\.\s+", line)
        or re.match(r"^#{1,6}\s", stripped)
        or stripped.startswith(("|", ">", "* ", "- "))
    )


def keeps_next_on_new_line(line: str) -> bool:
    return bool(
        re.match(r"^제\s*\d+\s*장\b", line)
        or re.match(r"^#{1,6}\s", line)
        or line.startswith("|")
    )


def join_wrapped_line(previous: str, current: str) -> str:
    if not previous:
        return current
    if not current:
        return previous

    prev_char = previous[-1]
    curr_char = current[0]
    if (
        re.match(r"[가-힣A-Za-z0-9]$", prev_char)
        and re.match(r"^[가-힣A-Za-z0-9]", curr_char)
    ) or curr_char in ".,;:)]}>%”’":
        return previous + current
    return previous + " " + current


def cleanup_spacing_artifacts(text: str) -> str:
    replacements = {
        "공공기관이이익": "공공기관이 이익",
        "관련된업무": "관련된 업무",
        "다음각 목": "다음 각 목",
        "및상임감사": "및 상임감사",
        "해당하는직무": "해당하는 직무",
        "소속기관을포함": "소속기관을 포함",
        "등에서유래": "등에서 유래",
        "명목에관계없이": "명목에 관계없이",
        "등의목적으로": "등의 목적으로",
        "가액범위안의": "가액범위 안의",
        "등이정하는": "등이 정하는",
        "범위에서일률적으로": "범위에서 일률적으로",
        "통하여받는": "통하여 받는",
        "친분관계를맺고": "친분관계를 맺고",
        "거쳐사장에게": "거쳐 사장에게",
        "10일이내": "10일 이내",
        "청탁방지담당관에게신고하고": "청탁방지담당관에게 신고하고",
        "받고수행하는": "받고 수행하는",
        "시직무관련성": "시 직무관련성",
        "강의내용이": "강의 내용이",
        "금지할수": "금지할 수",
        "별표 15에서정하는금액을초과하는사례금을받아서는아니된다": (
            "별표 15에서 정하는 금액을 초과하는 사례금을 받아서는 아니된다"
        ),
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return re.sub(r"별표\s*(\d+)", r"별표 \1", text)


def current_chapter(position: int, chapter_matches: list[re.Match]) -> tuple[str | None, str | None]:
    selected: re.Match | None = None
    for match in chapter_matches:
        if match.start() <= position:
            selected = match
        else:
            break
    if not selected:
        return None, None
    chapter = normalize_text(selected.group(0)).splitlines()[0].strip()
    chapter = normalize_chapter_title(chapter)
    number_match = re.match(r"(제\s*\d+\s*장)", chapter)
    chapter_number = normalize_text(number_match.group(1)) if number_match else None
    return chapter, chapter_number


def normalize_chapter_title(chapter: str) -> str:
    chapter = re.sub(r"(제\s*\d+\s*장)\s*보\s+칙\b", r"\1 보칙", chapter)
    chapter = re.sub(r"(제\s*\d+\s*장)\s*총\s+칙\b", r"\1 총칙", chapter)
    return chapter


def normalize_article_text(text: str) -> str:
    text = re.sub(r"(제\s*\d+\s*장)\s*보\s+칙\b", r"\1 보칙", text)
    text = re.sub(r"(제\s*\d+\s*장)\s*총\s+칙\b", r"\1 총칙", text)
    return text


def valid_article_matches(text: str) -> list[re.Match]:
    return [match for match in ARTICLE_RE.finditer(text) if is_article_heading_match(text, match)]


def is_article_heading_match(text: str, match: re.Match) -> bool:
    start, end = match.span()
    after = text[end:]
    next_nonspace = re.search(r"\S", after)
    next_char = next_nonspace.group(0) if next_nonspace else ""
    header = match.group(0)
    if "(" not in header and "<" not in header and next_char not in {"(", "<"}:
        return False
    if next_nonspace and next_nonspace.group(0) in {"제", "항", "호", "목"}:
        return False

    line_start = text.rfind("\n", 0, start) + 1
    line_prefix = text[line_start:start].strip()
    if line_prefix:
        return bool(re.search(r"(?:다|요|함|음|됨|\.|>)\s*\.?$", line_prefix))
    return True


def parse_article_heading(article_text: str) -> str:
    compact = re.sub(r"\s+", " ", article_text).strip()
    match = re.match(
        r"(제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\([^)]+\)|<[^>]+>)?)",
        compact,
    )
    if match:
        return normalize_text(match.group(1))
    return normalize_text(compact.split(" ", 1)[0])


def parse_article(article: str) -> tuple[str, str | None]:
    number_match = re.match(r"(제\s*\d+\s*조(?:의\s*\d+)?)", article)
    article_number = normalize_text(number_match.group(1)) if number_match else article

    paren_match = re.search(r"\(([^)]+)\)", article)
    if paren_match:
        return article_number, paren_match.group(1).strip()

    angle_match = re.search(r"<([^>]+)>", article)
    if angle_match:
        return article_number, angle_match.group(1).strip()

    return article_number, None


def build_appendix_chunk(
    document_name: str,
    source_file: Path,
    section: Section,
    page_spans: list[dict],
    config: dict,
    index: int,
) -> dict:
    raw = normalize_text(section_text_from_span(section, page_spans))
    raw = strip_angle_wrapped_heading(raw)
    first_line = first_non_empty_line(raw)
    title = strip_angle_wrapped_heading(section.title)
    if not title.startswith(("별표", "별지", "부칙")):
        title = first_line or title
    appendix = normalize_appendix_name(section.title)
    force_markdown = bool(config.get("force_appendix_markdown")) and re.sub(r"\s+", "", appendix).startswith("별표")
    table_like = (force_markdown or has_table_shape(raw)) and not re.sub(r"\s+", "", appendix).startswith("부칙")
    content = to_markdown_table_like(raw, title) if config.get("table_to_markdown") and table_like else raw
    content = cleanup_spacing_artifacts(format_nested_letter_items(content))
    title = appendix_title_from_content(content, title)
    content_format = "markdown" if config.get("table_to_markdown") and table_like else "text"
    is_table = content_format == "markdown" and table_like
    section_type = "appendix_table" if is_table else "appendix"
    page_start, page_end = pages_for_span(section.start, section.end, page_spans)

    return {
        "chunk_id": generate_chunk_id(
            document_name,
            section_type,
            appendix=appendix,
            index=index,
        ),
        "document_name": document_name,
        "source_file": normalize_filename(source_file),
        "section_type": section_type,
        "chapter": None,
        "chapter_number": None,
        "article": None,
        "article_number": None,
        "article_title": None,
        "appendix": appendix,
        "title": title,
        "content": content,
        "content_format": content_format,
        "page_start": page_start,
        "page_end": page_end,
        "metadata": {
            "document_type": "pdf",
            "chunking_unit": section_type,
            "is_appendix": True,
            "is_table": is_table,
            "is_deleted_article": False,
            "table_format": "markdown" if is_table else None,
            "structure_key": f"{document_name} > {title}",
        },
    }


def section_text_from_span(section: Section, page_spans: list[dict]) -> str:
    return section.text


def normalize_appendix_name(title: str) -> str:
    title = strip_angle_wrapped_heading(title)
    title = re.sub(r"\s+", " ", title).strip()
    match = re.match(r"별표\s*(\d+)", title)
    if match:
        return f"별표 {match.group(1)}"
    match = re.match(r"(별지\s*제?\s*\d*\s*호)", title)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    if re.match(r"부\s*칙", title):
        return "부칙"
    return title


def appendix_title_from_content(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        match = re.match(r"^#{1,6}\s*(별표\s*\d+)\s*(.*)$", stripped)
        if match:
            appendix = f"별표 {match.group(1).replace('별표', '').strip()}"
            suffix = match.group(2).strip()
            return f"{appendix} {suffix}".strip()

        match = re.match(r"^(별표\s*\d+)\s*(.*)$", stripped)
        if match:
            appendix = f"별표 {match.group(1).replace('별표', '').strip()}"
            suffix = match.group(2).strip()
            return f"{appendix} {suffix}".strip()

    return fallback


def strip_angle_wrapped_heading(text: str) -> str:
    text = re.sub(r"^\s*<([^>\n]+)>\s*", r"\1\n", text).strip()
    return re.sub(r"^([^<>\n]+)>\s*", r"\1\n", text).strip()


def has_table_shape(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    slash_headers = any("/" in line and len(line.split("/")) >= 3 for line in lines[:5])
    numbered_rows = sum(1 for line in lines if re.match(r"^\d+[\s.)]", line))
    wide_rows = sum(1 for line in lines if len(re.split(r"\s{2,}|\t", line)) >= 3)
    return slash_headers or numbered_rows >= 2 or wide_rows >= 2


def to_markdown_table_like(text: str, title: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    heading = title if title.startswith(("별표", "별지")) else lines[0]
    body_lines = lines[1:] if lines and lines[0] == heading else lines
    special = format_known_kotra_appendix(text, heading)
    if special:
        return special

    markdown = [f"### {heading}", ""]

    table = try_parse_simple_table(body_lines)
    if table:
        header, rows = table
        markdown.append("| " + " | ".join(header) + " |")
        markdown.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            padded = row + [""] * (len(header) - len(row))
            markdown.append("| " + " | ".join(padded[: len(header)]) + " |")
        return "\n".join(markdown)

    grouped_rows = group_numbered_rows(body_lines)
    if grouped_rows:
        for number, content in grouped_rows:
            markdown.append(f"- 순번: {number}")
            if content:
                markdown.append(f"  - 내용: {content}")
        return "\n".join(markdown)

    for line in body_lines:
        if line == heading:
            continue
        markdown.append(f"- {line}")
    return "\n".join(markdown)


def format_known_kotra_appendix(text: str, heading: str) -> str | None:
    key = re.sub(r"\s+", "", heading)
    cleaned = cleanup_appendix_text(text)
    if key.startswith("별표11"):
        return format_appendix_11(cleaned)
    if key.startswith("별표14"):
        return format_appendix_14(cleaned)
    if key.startswith("별표15"):
        return format_appendix_15(cleaned)
    if key.startswith("별표21"):
        return format_appendix_21(cleaned)
    if key.startswith("별표28"):
        return format_appendix_28()
    if key.startswith("별표1"):
        return format_appendix_1()
    return None


def cleanup_appendix_text(text: str) -> str:
    text = strip_angle_wrapped_heading(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\n\s*<\s*\n", "\n", text)
    text = re.sub(r"(?m)^\s*<별표\s*\d+>\s*$", "", text)
    text = re.sub(r"(?m)^\s*<별표\d+>\s*$", "", text)
    text = re.sub(r"(?m)^\s*순번\s+부서\s+업무명\s+청탁유형\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_appendix_1() -> str:
    rows = [
        ("1", "홍보실", "언론대응", "공사에 우호적인 언론기사 작성을 목적으로 과도한 접대·향응 등 제공 요구"),
        ("2", "홍보실", "홍보비 집행", "특정 업체에 광고비를 집중해서 집행하도록 청탁"),
        ("3", "기획조정실", "예산편성 및 집행", "예산편성 및 집행 시 특정사업이나 조직에 편파적으로 지원 요구"),
        ("4", "기획조정실", "조직운영", "본사 및 조직망 조직운영 시 특정사업 또는 조직에 인력을 편파적으로 지원하도록 요구"),
        ("5", "기획조정실", "내부평가", "평가 목표부여 및 조정 시 특정 조직이 유리하도록 특혜 제공 요구; 부당한 방법으로 높은 평가등급 요구; 특정부서를 유리하거나 불리하게 할 목적으로 평가결과 조작 요구"),
        ("6-1", "운영지원실", "사옥관리", "시설관리 용역업체 선정 시 특정업체가 선정되도록 청탁"),
        ("6-2", "운영지원실", "계약", "입찰정보 사전 제공 요구; 특정업체에 유리한 수의계약 특혜 제공 요구; 계약 관련 특정인 선정 요구; 계약 관련 평가위원 선정 시 특정인 선정 요구; 물품단가 인상 요구; 계약 진행 시 특정업체 하자 또는 과실 간과 요구"),
        ("7", "운영지원실", "자금운영", "자금조달 또는 여유자금 운영 시 특정 금융기관에 특혜 제공 요구; 주거래 은행 선정 시 특정은행에 유리하도록 요구"),
        ("8", "운영지원실", "사택관리", "부적절한 비품, 가전제품, 인테리어 요구"),
        ("9", "인재경영실", "채용", "신입, 경력, 전문직 등 직원 채용 시 공정성을 저해하는 청탁"),
        ("10", "인재경영실", "승진 및 전보", "직원 승진 및 전보 시 특정인에게 유리하도록 특혜 제공 요구"),
        ("11", "인재경영실", "교육", "직원 교육연수 업체로 선정되기 위해 담당자에게 청탁; 국내외 위탁교육 대상자로 선정되기 위해 담당자에게 청탁"),
        ("12", "인재경영실", "인사정보", "사적인 용도로 타인의 개인정보 요구"),
        ("13-1", "정보시스템팀", "전산기기 도입", "전산기기 기종 선정 및 도입 시 특정 업체 물품 선정 청탁; 특정 S/W 또는 H/W 도입 요구"),
        ("13-2", "정보시스템팀", "전산시스템 개발·운영", "업무상 필요나 개선 효과가 없는 시스템 개발 청탁 의뢰"),
        ("14", "사업부서", "수출지원업무", "사업 우선 참여 요구; 지사화사업, 전시회 등 사업 참가를 위한 유리한 시장성평가 청탁; 무리한 사업 참가 선정 요구; 전시회 등 대형행사 대행사 선정 관련 특정업체 선정 요구; 수의계약 체결 시 특정인 계약 당사자 선정·탈락 개입 요구"),
        ("15", "사업부서", "투자유치업무", "외국인투자유치포상금 관련 청탁"),
        ("16", "감사실", "감사업무", "감사결과 지적사항 무마 요구; 징계처분 감경 요구; 특정인 비방 목적의 감사 의뢰; 신고자 정보 제공 요구"),
        ("17", "해외조직망", "업무전반", "자동차 등 각종 자산 구입 관련 특정 업체의 부당한 물품 구매 요구; 고객의 부당한 서비스 청탁; 현지직원 채용 관련 청탁"),
        ("18", "공통", "업무전반", "인가·허가·면허·검정·시험·인증·확인 등의 직무수행 처리 요구; 위원·시험·선발위원 등 직위 선정 또는 탈락 요구; 수상·포상자, 우수기관/기업, 우수자 선정 개입 요구; 직무상 비밀 누설 요구; 각종 평가·판정 업무 결과 조작 요구"),
    ]
    return markdown_table(
        "별표 1 부서별 청탁등록 대상 업무 및 청탁유형",
        ["순번", "부서", "업무명", "청탁유형"],
        rows,
    )


def format_appendix_11(text: str) -> str:
    text = text.replace("\n<\n", "\n")
    return "\n".join(
        [
            "### 별표 11 부당한 업무지시의 판단기준",
            "",
            "1. 판단기준",
            "",
            "   가. 법령, 행정규칙(훈령·예규·고시·지침 등), 사규에 위반되는 지시인지 여부",
            "   나. 업무의 본래 취지에 맞지 않는 지시인지 여부",
            "   다. 공공기관에 재산상 손해를 입힐 수 있는 지시인지 여부",
            "   라. 공적이익이 아닌 사적이익을 추구하는 지시인지 여부",
            "   마. 지위 또는 권한을 남용하는 지시인지 여부",
            "   바. 자율성이 보장된 것임에도 행위를 강요하는 지시인지 여부",
            "   사. 그 밖에 현저히 불합리한 행위를 강제하는 지시인지 여부",
            "",
            "2. 부당한 업무지시에 해당될 수 있는 유형 예시",
            "",
            "> 자기 또는 타인의 부당한 이익을 위하여 다음 유형의 지시를 할 경우 부당지시에 해당될 수 있음",
            "",
            "   가. 규정위반 내용 또는 본래의 취지에 맞지 않는 방향으로 지시",
            "   나. 신고사건 등 민원처리에 개입하여 부당하게 방향을 지시",
            "   다. 신고사건 처리 시 필요 이상으로 상위자를 출석요구토록 지시",
            "   라. 점검 등 계획수립 시 합리적인 이유 없이 특정업체를 포함 또는 제외토록 지시",
            "   마. 관용차 등 공용물을 휴일 등에 사적용도로 사용하기 위한 지시",
            "   바. 물품구매 등 각종 계약 시 정당한 이유 없이 특정업체를 선정토록 지시",
            "   사. 업무추진비 등 예산을 사적용도로 집행토록 지시",
            "   아. 인사에 있어 지연·혈연·학연·직연 등 비합리적인 연고성·편파적 운영 지시",
            "   자. 근무성적 평가를 이유로 협박성 회유 또는 부당한 지시",
            "   차. 직원에게 직무관련자를 통하여 골프부킹, 콘도예약 등 부당한 지시",
            "   카. 직무관련자에게 취업을 청탁하도록 지시",
            "   타. 개인적 경조사를 직무관련자에게 알리도록 지시",
            "   파. 사업장 등에 자신의 외부강의를 주선하도록 지시",
        ]
    )


def format_appendix_14(text: str) -> str:
    return "\n".join(
        [
            "### 별표 14 음식물·경조사비·선물 등의 가액 범위",
            "",
            "1. 음식물(제공자와 공직자등이 함께 하는 식사, 다과, 주류, 음료, 그 밖에 이에 준하는 것을 말한다) : 5만원 <개정 2024.08.27>",
            "",
            "2. 경조사비 : 축의금·조의금은 5만원. 다만, 축의금·조의금을 대신하는 화환·조화는 10만원.",
            "",
            "3. 선물 : 다음 각 목의 금품등을 제외한 일체의 물품 및 상품권(물품상품권 및 용역상품권만 해당하며, 이하 “상품권”이라 한다), 그 밖에 이에 준하는 것은 5만원. 다만, 「농수산물 품질관리법」 제2조제1항제1호에 따른 농수산물(이하 “농수산물”이라 한다) 및 같은 항 제13호에 따른 농수산가공품(농수산물을 원료 또는 재료의 50퍼센트를 넘게 사용하여 가공한 제품만 해당하며, 이하 “농수산가공품”이라 한다)과 농수산물ㆍ농수산가공품 상품권은 15만원(제17조제2항에 따른 기간 중에는 30만원)으로 한다.",
            "",
            "   가. 금전",
            "",
            "   나. 유가증권(상품권은 제외한다)",
            "",
            "   다. 제1호의 음식물",
            "",
            "   라. 제2호의 경조사비",
            "",
            "#### 비고",
            "",
            "가. 제1호, 제2호 본문·단서 및 제3호 본문·단서의 각각의 가액 범위는 각각에 해당하는 것을 모두 합산한 금액으로 한다.",
            "",
            "나. 제2호 본문의 축의금·조의금과 같은 호 단서의 화환·조화를 함께 받은 경우 또는 제3호 본문의 선물과 같은 호 단서의 농수산물·농수산가공품을 함께 받은 경우에는 각각 그 가액을 합산한다. 이 경우 가액 범위는 10만원으로 하되, 제2호 본문 또는 단서나 제3호 본문 또는 단서의 가액 범위를 각각 초과해서는 안된다.",
            "",
            "다. 제3호의 상품권이란 그 명칭 또는 형태에 관계없이 발행자가 특정한 물품 또는 용역의 수량을 기재(전자적 또는 자기적 방법에 의한 기록을 포함한다)하여 발행ㆍ판매하고, 그 소지자가 발행자 또는 발행자가 지정하는 자(이하 “발행자등”이라 한다)에게 이를 제시 또는 교부하거나 그 밖의 방법으로 사용함으로써 그 증표에 기재된 내용에 따라 발행자등으로부터 해당 물품 또는 용역을 제공받을 수 있는 증표인 물품상품권 또는 용역상품권을 말하며, 백화점상품권ㆍ온누리상품권ㆍ지역사랑상품권ㆍ문화상품권 등 일정한 금액이 기재되어 소지자가 해당 금액에 상응하는 물품 또는 용역을 제공받을 수 있는 증표인 금액상품권은 제외한다.",
            "",
            "라. 제3호 본문의 선물과 같은 호 단서의 농수산물ㆍ농수산가공품 또는 농수산물ㆍ농수산가공품 상품권을 함께 받은 경우에는 그 가액을 합산한다. 이 경우 가액 범위는 15만원(제17조제2항에 따른 기간 중에는 30만원)으로 하되, 제3호 본문 또는 단서의 가액 범위를 각각 초과해서는 안 된다.",
            "",
            "마. 제1호의 음식물, 제2호의 경조사비 및 제3호의 선물 중 2가지 이상을 함께 받은 경우에는 그 가액을 합산한다. 이 경우 가액 범위는 함께 받은 음식물, 경조사비 및 선물의 가액 범위 중 가장 높은 금액으로 하되, 제1호부터 제3호까지의 규정에 따른 가액 범위를 각각 초과해서는 안 된다.",
        ]
    )


def format_appendix_15(text: str) -> str:
    return markdown_table(
        "별표 15 외부강의 등 사례금 상한액",
        ["구분", "기준(천원)"],
        [("최초 1시간(상한액)", "400"), ("1시간 초과", "200")],
    ) + "\n\n" + "\n".join(
        [
            "* 국제기구, 외국정부, 외국대학, 외국연구기관, 외국학술단체, 그 밖에 이에 준하는 외국기관에서 지급하는 외부강의 등의 사례금 상한액은 사례금을 지급하는 자의 지급기준에 따른다.",
            "* 상한액은 강의 등의 경우 1시간당, 기고의 경우 1건당 상한액으로 한다.",
            "* 1시간을 초과하여 강의 등을 하는 경우에도 사례금 총액은 강의시간에 관계없이 1시간 상한액의 100분의 150에 해당하는 금액을 초과하지 못한다.",
            "* 상한기준은 강의료, 원고료, 출연료 등 명목에 관계없이 사례금 제공자가 제공하는 일체의 사례금을 포함하며, 강의가 원격지에서 시행되어 교통비, 일식비, 숙박비 등 출장여비를 제공받을 필요가 있을 경우 외부강의 등 사례금 제공자로부터 「공무원 여비규정」 등 공공기관별로 적용되는 여비 규정의 기준 내에서 실비수준으로 제공되는 교통비, 숙박비 및 식비를 별도로 수령할 수 있다.",
        ]
    )


def format_appendix_21(text: str) -> str:
    return "\n".join(
        [
            "### 별표 21 고객 접촉 시 청렴수칙",
            "",
            "① 고객으로부터 금품(선물, 주류, 상품권 등) 받지 않기",
            "",
            "* 가능하면 그 자리에서 거절하는 것이 최선",
            "* 거절하기 어려운 상황(예: 부재 시 두고 갔을 경우 등)이었으면, 즉시 반환 (반환비용 청구 가능)",
            "* 주소 불명 등으로 반환하기 어려운 경우 감사실에 신고",
            "",
            "② 고객으로부터 향응 (3만원 이상의 식사, 접대 등) 받지 않기",
            "",
            "* 고객이 접대하는 식사, 술자리 등 부담스러운 자리는 되도록 피할 것",
            "",
            "③ 고객에게서 편의(숙박시설, 교통편의, 행사 협찬, 부적절한 업무지원 등) 받지 않기",
            "",
            "* 고객이 친절을 베풀더라도 사양할 것",
            "* 고객이 무역관 지사화 전담직원의 동행출장 요청 시 숙박비, 교통임을 부담토록 하는 것은 공사의 내부규정에 의한 것임을 명확히 주지",
            "",
            "④ 업무처리 시 기준이나 절차를 투명하게 하고 공개하기",
            "",
            "* 진행사항 애로사항 등 고객에게 주기적으로 보고",
            "* 사업 거절 시 사유 및 근거 제시",
            "",
            "⑤ 사업수행 시 기한을 준수하고 고객과의 약속 지키기",
            "",
            "⑥ 지연·학연·혈연관계에 따라 특정인에게 혜택주지 않기",
        ]
    )


def format_appendix_28() -> str:
    rows = [
        ("신입사원", "임직원 행동강령 등 규정교육, 부패 취약 업무에 대한 신규 입사자 대응 방안, 임직원의 바람직한 근무자세 등", "기관장 및 행동강령책임관 특별교육, 교육훈련기관 집합교육 및 사이버교육 등", "-입사 후 1년 이내<br>-연간 2시간 이상", ""),
        ("해외 발령자 및 귀임자", "임직원 행동강령 등 윤리규범교육, 해외무역관 부패 취약 업무에 대한 대응방안, 근무환경 변화에 따른 적응방안", "기관장 및 행동강령책임관 특별교육, 교육훈련기관 집합교육 및 사이버교육 등", "해외발령전 및 귀임후 2개월 이내", ""),
        ("승진(예정)자", "중간관리자의 바람직한 근무자세 및 역할, 주요 부패사례 및 슬기로운 대처법, 청렴리더십 확립 등", "기관장 특별교육, 교육훈련기관 집합교육 및 사이버교육 등", "승진 전후 1년이내<br>연간 5시간 이상", ""),
        ("고위간부", "노블레스 오블리주 함양, 공직자의 사회공헌, 청렴리더십 확립 등", "감사 특별교육, 사이버교육 등", "분기별 1회", ""),
    ]
    return markdown_table(
        "별표 28 공직생애 주기별 청렴교육 의무이수기준",
        ["교육대상", "교육분야", "교육방법", "교육시간", "비고"],
        rows,
    )


def markdown_table(title: str, headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [escape_markdown_table_cell(str(cell)) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def escape_markdown_table_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|").strip()


def try_parse_simple_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    candidates: list[list[str]] = []
    for line in lines:
        if "/" in line:
            cells = [cell.strip() for cell in line.split("/") if cell.strip()]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}|\t", line) if cell.strip()]
        if len(cells) >= 3:
            candidates.append(cells)

    if len(candidates) < 2:
        return None
    width = max(3, min(len(candidates[0]), 6))
    header = candidates[0][:width]
    rows = [row[:width] for row in candidates[1:] if len(row) >= 2]
    if not rows:
        return None
    return header, rows


def group_numbered_rows(lines: list[str]) -> list[tuple[str, str]]:
    groups: list[tuple[str, list[str]]] = []
    current_number: str | None = None
    current_parts: list[str] = []

    for line in lines:
        if re.search(r"순번|구분", line) and not re.match(r"^\d+\s+", line):
            continue
        match = re.match(r"^(\d+)\s*(.*)", line)
        if match:
            if current_number is not None:
                groups.append((current_number, current_parts))
            current_number = match.group(1)
            remainder = match.group(2).strip()
            current_parts = [remainder] if remainder else []
            continue
        if current_number is not None:
            current_parts.append(line)

    if current_number is not None:
        groups.append((current_number, current_parts))

    if len(groups) < 2:
        return []
    return [(number, " ".join(part for part in parts if part).strip()) for number, parts in groups]
