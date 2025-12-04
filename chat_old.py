# chat.py

import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# --- 기본 설정 ---
APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="무역관 정산 챗봇", page_icon="💰")
st.title("💰무역관 정산 챗봇")
st.caption("해외무역관 정산에 대한 모든 것을 물어보세요!")

load_dotenv()

# --- 이전 대화 기록 표시 ---
if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "ai" and message.get("source"):
            st.caption(message["source"])

# --- 사용자 입력 처리 ---
if user_question := st.chat_input("해외무역관 정산에 대한 궁금한 내용을 물어보세요"):
    st.session_state.message_list.append({"role": "user", "content": user_question})
    st.session_state.message_list.append({"role": "ai", "content": "", "source": None})
    st.rerun()

# --- 스트리밍 실행 ---
if (
    st.session_state.message_list
    and st.session_state.message_list[-1]["role"] == "ai"
    and st.session_state.message_list[-1]["content"] == ""
):
    user_question = st.session_state.message_list[-2]["content"]

    # ---- page_num 포맷 함수 ----
    def _format_page_token(page_val):
        """
        포맷 규칙:
        - 숫자(int) 또는 '7', '7.0', '07.00' 등 숫자 형태 → '7p'
        - 문자 섞인 경우 원문 그대로 유지 ('2조 목적' 등)
        """
        if page_val is None:
            return None

        # int → '3p'
        if isinstance(page_val, int):
            return f"{page_val}p"

        s = str(page_val).strip()
        if not s:
            return None

        # float 형태 문자열 ('7.0', '3.00') → int로 변환 후 p
        try:
            num = float(s)
            # 정수형(float지만 .0으로 끝나는 경우)
            if num.is_integer():
                return f"{int(num)}p"
        except ValueError:
            pass

        # 순수 숫자 문자열 ('12') → '12p'
        if re.fullmatch(r"\d+", s):
            return f"{s}p"

        # 문자 섞임 ('2조 목적', 'p.12' 등) → 그대로
        return s

    # ---- 순서 보존 중복제거 ----
    def _dedup_preserve_order(items):
        seen = set()
        out = []
        for x in items:
            if x is None:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # ---- 파일명 정제 ----
    def _resolve_pdf_name(metadata: dict):
        name = (metadata or {}).get("origin_pdf") or (metadata or {}).get("source") or "알 수 없음"
        try:
            return Path(str(name)).name
        except Exception:
            return str(name)

    with st.spinner("답변을 생성하는 중입니다..."):
        ai_response_stream = get_ai_response(user_question)

        full_answer = ""
        source_info = None

        # 문서별 페이지 수집 구조
        origins_order = []
        pages_by_origin = {}
        seen_pairs = set()

        for chunk in ai_response_stream:
            # ---- 컨텍스트(문서)에서 출처 정보 수집
            if "context" in chunk and source_info is None:
                docs = chunk["context"] or []
                for d in docs:  
                    md = getattr(d, "metadata", {}) or {}
                    origin = _resolve_pdf_name(md)
                    page_token = _format_page_token(md.get("page_num"))
                    if not page_token:
                        continue

                    pair = (origin, page_token)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    if origin not in pages_by_origin:
                        pages_by_origin[origin] = []
                        origins_order.append(origin)

                    if page_token not in pages_by_origin[origin]:
                        pages_by_origin[origin].append(page_token)

                # ---- 표기 문자열 구성
                if origins_order:
                    parts = []
                    for origin in origins_order:
                        pages = _dedup_preserve_order(pages_by_origin.get(origin, []))
                        if not pages:
                            continue
                        # 여러 페이지면 "a, b (파일명)", 하나면 "a (파일명)"
                        pages_str = ", ".join(pages)
                        parts.append(f"{pages_str} ({origin})")
                    if parts:
                        source_info = "📄출처: " + ", ".join(parts)
                    else:
                        source_info = "📄출처: 페이지 정보 없음"
                else:
                    source_info = "📄출처: 페이지 정보 없음"

            # ---- 답변 스트림 이어붙이기
            if "answer" in chunk:
                full_answer += chunk["answer"]

        # ---- 세션 업데이트
        st.session_state.message_list[-1]["content"] = full_answer
        st.session_state.message_list[-1]["source"] = source_info
        st.rerun()
