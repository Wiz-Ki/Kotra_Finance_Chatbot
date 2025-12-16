# chat.py

import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response


# --- 기본 설정 ---
APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="무역관 정산 챗봇", page_icon="💰")

# --------------------------------------------------------------------------
# [추가됨] 비밀번호 확인 함수
# --------------------------------------------------------------------------
def check_password():
    """비밀번호가 맞는지 확인하여 True/False를 반환하는 함수"""
    
    # 1. 세션에 비밀번호 확인 여부 변수가 없으면 초기화
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # 2. 이미 비밀번호를 맞췄다면 True 반환 (바로 접속)
    if st.session_state["password_correct"]:
        return True

    # 3. 비밀번호 입력 화면 출력
    st.header("🔒 접근 제한")
    st.write("무역관 정산 챗봇에 접속하려면 비밀번호를 입력하세요.")
    
    password_input = st.text_input("비밀번호", type="password")

    if password_input:
        # Streamlit Cloud의 Secrets에 저장된 "PASSWORD"와 비교
        if password_input == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun()  # 비밀번호가 맞으면 화면을 새로고침하여 앱 실행
        else:
            st.error("비밀번호가 틀렸습니다. 다시 시도해주세요.")

    return False

# --------------------------------------------------------------------------
# [추가됨] 로그인 체크 실행
# 비밀번호가 확인되지 않으면 여기서 코드 실행을 멈춥니다 (st.stop)
# --------------------------------------------------------------------------
if not check_password():
    st.stop()
# --------------------------------------------------------------------------
    
st.title("💰무역관 정산 챗봇")
st.caption("더 정확한 답변을 위해 사우님들의 피드백이 필요해요 🌱 답변 우측 하단의 [좋아요👍/싫어요👎] 꼭 눌러주세요!")

load_dotenv()

# --- 이전 대화 기록 표시 ---
if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "ai" and message.get("source"):
            st.caption(message["source"])

# --------------------------------------------------------------------------
# [수정됨] 안내 문구 및 레이아웃 수정
# 1. 입력창이 내용을 가리지 않도록 본문 하단에 여백(Padding) 추가
# 2. 입력창을 위로 올리고, 그 아래에 안내 문구 배치
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* [핵심] 본문 내용이 가려지지 않도록 하단 여백 확보 */
        /* 입력창 높이(약 50px) + 안내문구 높이(50px) + 여유분 */
        .main .block-container {
            padding-bottom: 120px !important;
        }

        /* 1. 채팅 입력창(stChatInput) 디자인 및 위치 조정 */
        [data-testid="stChatInput"] {
            bottom: 50px !important; /* 안내 문구 높이만큼 위로 띄움 */
            background-color: transparent !important; /* 배경색 투명하게 */
        }
        
        /* (선택사항) 입력창 주변의 붕 뜬 그림자나 경계선 제거가 필요하면 추가 */
        [data-testid="stChatInput"] > div {
            border-color: transparent !important;
        }

        /* 2. 안내 문구 영역 (화면 최하단 고정) */
        .footer-disclaimer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 95px;
            background-color: #ffffff; /* 배경을 흰색으로 채워서 뒤가 비치지 않게 함 */
            color: #888888;
            text-align: center;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100; /* 맨 위에 표시 */
            border-top: 1px solid #f0f0f0; /* 상단에 얇은 구분선 */
        }
    </style>

    <div class="footer-disclaimer">
        <div>
            저는 아직 배우는 중이라 실수가 있을 수 있어요! 😅 <br>
            답변은 참고만 해주시고, 헷갈리는 부분은 꼭 재무팀 담당자분들께 확인 부탁드려요.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# --------------------------------------------------------------------------

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
                        # 여러 페이지면 "(파일명) a, b", 하나면 "(파일명) a"
                        pages_str = ", ".join(pages)
                        # ✅ 문서명이 먼저 나오고, 그 뒤에 페이지
                        parts.append(f"({origin}) {pages_str}")
                    if parts:
                        # ✅ 문서 사이 구분자를 ", " 대신 " / "로 변경
                        source_info = "📄출처: " + " / ".join(parts)
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
