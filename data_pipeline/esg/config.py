"""Configuration for KOTRA ethics and integrity PDF chunking."""

from __future__ import annotations


DOCUMENT_CONFIG = {
    "공직자의 이해충돌 방지법": {
        "unit": "장-조",
        "has_chapter": True,
        "include_appendix": False,
        "drop_sections": ["부칙", "부 칙"],
        "include_deleted_articles": True,
        "keep_deleted_articles_as_chunks": True,
        "table_to_markdown": False,
        "source_keywords": ["공직자의", "이해충돌", "방지법"],
    },
    "부정청탁 및 금품등 수수의 금지에 관한 법률": {
        "unit": "장-조",
        "has_chapter": True,
        "include_appendix": False,
        "drop_sections": ["부칙", "부 칙"],
        "include_deleted_articles": True,
        "keep_deleted_articles_as_chunks": True,
        "table_to_markdown": False,
        "source_keywords": ["부정청탁", "금품", "수수", "금지"],
    },
    "직장 내 인권침해 예방지침": {
        "unit": "조",
        "has_chapter": False,
        "include_appendix": False,
        "drop_sections": ["부칙", "부 칙", "별지", "별지 제"],
        "include_deleted_articles": True,
        "keep_deleted_articles_as_chunks": True,
        "table_to_markdown": False,
        "source_keywords": ["직장", "인권침해", "예방지침"],
    },
    "KOTRA 임직원 이해충돌 방지제도 운영지침": {
        "unit": "조",
        "has_chapter": False,
        "include_appendix": False,
        "appendix_type": "별지",
        "drop_sections": ["부칙", "부 칙", "별지", "별지 제"],
        "include_deleted_articles": True,
        "keep_deleted_articles_as_chunks": True,
        "table_to_markdown": True,
        "source_keywords": ["KOTRA", "임직원", "이해충돌", "방지제도", "운영지침"],
    },
    "KOTRA 임직원 행동강령": {
        "unit": "장-조",
        "has_chapter": True,
        "include_appendix": True,
        "appendix_type": "별표/별지",
        "drop_sections": ["부칙", "부 칙"],
        "include_deleted_articles": True,
        "keep_deleted_articles_as_chunks": True,
        "table_to_markdown": True,
        "force_appendix_markdown": True,
        "include_appendix_names": ["별표1", "별표11", "별표14", "별표15", "별표21", "별표28"],
        "appendix_aliases": [
            {
                "title": "별표 11 부당한 업무지시의 판단기준",
                "pattern": r"(?:^|\n)\s*(부당한 업무지시의 판단기준)\s*(?:\n|$)",
            }
        ],
        "source_keywords": ["KOTRA", "임직원", "행동강령"],
    },
}


CHAPTER_PATTERN = r"제\s*\d+\s*장\s*[^\n]*"

ARTICLE_PATTERN = (
    r"제\s*\d+\s*조"
    r"(?:의\s*\d+)?"
    r"\s*"
    r"(?:\([^)]+\)|<[^>]+>)?"
)

APPENDIX_PATTERN = (
    r"^(부칙|부\s*칙|별표\s*\d*|별지\s*제?\s*\d*\s*호)"
)

DELETED_ARTICLE_PATTERN = (
    r"제\s*\d+\s*조(?:의\s*\d+)?\s*<본조\s*삭제[^>]*>"
)

APPENDIX_INLINE_PATTERN = (
    r"(?:^|\n)\s*<?("
    r"별표\s*\d+|"
    r"별지\s*제?\s*\d+\s*호(?:\s*서식)?|"
    r"부\s*칙(?:\s*\([^)]+\))?"
    r")>?"
)
