# chat.py

import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# --- 경로 설정: chat.py가 있는 폴더/이미지 폴더 ---
APP_DIR = Path(__file__).resolve().parent
IMG_DIR = APP_DIR / "md_images"

st.set_page_config(page_title="무역관 정산 챗봇", page_icon="💰")
st.title("💰무역관 정산 챗봇")
st.caption("해외무역관 정산에 대한 모든 것을 물어보세요!")

load_dotenv()

# --- 유틸: 이미지 경로 변환/정렬, 중복 제거(순서 보존) ---
def _resolve_image_paths(img_list):
    """상대 파일명을 md_images/ 경로로 변환"""
    paths = []
    for name in img_list or []:
        name = str(name).strip().lstrip("/\\")
        p = (IMG_DIR / name)
        paths.append(p.as_posix())
    return paths

def _natural_sort_key(s: str):
    """파일명 안 숫자 기준 자연스러운 정렬 키"""
    nums = re.findall(r'\d+', s or "")
    return [int(n) for n in nums] if nums else [float('inf')]

def _dedup_preserve_order(items):
    """중복 제거하되, 최초 등장 순서를 보존"""
    seen = set()
    result = []
    for x in items or []:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

# --- 이전 대화 기록과 출처/이미지 함께 출력 ---
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "ai" and "source" in message and message["source"]:
            st.caption(message["source"])
        # ✅ AI 메시지에 이미지가 있으면 실제 이미지로 렌더
        if message["role"] == "ai":
            for img_path in _resolve_image_paths(message.get("images", [])):
                st.image(img_path, width='stretch')  # use_container_width → width='stretch'

# --- 사용자 입력 처리 ---
if user_question := st.chat_input(placeholder="해외무역관 정산에 대한 궁금한 내용을 물어보세요"):
    st.session_state.message_list.append({"role": "user", "content": user_question})
    st.session_state.message_list.append({"role": "ai", "content": "", "source": None, "images": []})
    st.rerun()

# --- 스트리밍 실행 조건 ---
if (
    st.session_state.message_list
    and st.session_state.message_list[-1]["role"] == "ai"
    and st.session_state.message_list[-1]["content"] == ""
):
    user_question = st.session_state.message_list[-2]["content"]

    with st.spinner("답변을 생성하는 중입니다..."):
        ai_response_stream = get_ai_response(user_question)

        full_answer = ""
        source_info = None
        collected_images = []  # ✅ 유사도 순서(문서 순서)를 유지하여 모음

        # 스트리밍 처리
        for chunk in ai_response_stream:
            if "context" in chunk and source_info is None:
                docs = chunk["context"]
                if docs:
                    # 1) origin_pdf는 가장 유사한 문서(첫 문서) 기준으로 1개만
                    first_doc = docs[0]
                    pdf_name = getattr(first_doc, "metadata", {}).get("origin_pdf", "없음")

                    # 2) 페이지 번호: '유사도 순서' 유지 (정렬 X), 중복만 제거
                    page_nums_in_rank_order = []
                    for d in docs[:2]:
                        md = getattr(d, "metadata", {}) or {}

                        # 페이지 번호 수집 (유사도 순서 유지)
                        p = md.get("page_num")
                        try:
                            p = int(p)
                            if p not in page_nums_in_rank_order:
                                page_nums_in_rank_order.append(p)
                        except (TypeError, ValueError):
                            pass

                        # 3) 이미지 수집:
                        #    - 문서 내부는 natural sort
                        #    - 문서들 간 전체 순서는 '유사도 순' 유지
                        imgs = md.get("images") or []
                        imgs = [im.strip() for im in imgs if isinstance(im, str) and im.strip()]
                        imgs = sorted(imgs, key=_natural_sort_key)
                        collected_images.extend(imgs)

                    # 4) 출처 포맷: '8p, 3p / 파일명.pdf' (유사도 순서 유지)
                    pages_str = ", ".join(f"{p}p" for p in page_nums_in_rank_order)
                    source_info = f"📄 출처: {pages_str} / {pdf_name}"

            if "answer" in chunk:
                full_answer += chunk["answer"]

        # 5) 이미지 중복 제거(순서 보존: 유사도 높은 문서의 이미지가 앞에 오도록)
        collected_images = _dedup_preserve_order(collected_images)

        # 세션 업데이트
        st.session_state.message_list[-1]["content"] = full_answer
        st.session_state.message_list[-1]["source"] = source_info
        st.session_state.message_list[-1]["images"] = collected_images

        st.rerun()
