import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

# 구글 시트 및 피드백 관련 라이브러리
from streamlit_feedback import streamlit_feedback
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import uuid
import pytz # [필수] 시간대 처리를 위해 추가

# --- 기본 설정 ---
APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="무역관 정산 챗봇", page_icon="💰")

# --------------------------------------------------------------------------
# [설정] 무역관별 시간대 매핑 (필요한 만큼 추가하세요)
# --------------------------------------------------------------------------
# IANA Timezone Database 이름을 사용합니다.
BRANCH_TIMEZONES = {
    "뉴욕무역관": "America/New_York",
    "도쿄무역관": "Asia/Tokyo",
    "상하이무역관": "Asia/Shanghai",
    "프랑크푸르트무역관": "Europe/Berlin",
    "런던무역관": "Europe/London",
    "본사": "Asia/Seoul",
    "기타": "Asia/Seoul",
    "테스트": "Asia/Seoul" # 기본값
}

# --------------------------------------------------------------------------
# [기능 1] 구글 시트 연동 함수
# --------------------------------------------------------------------------
@st.cache_resource
def get_google_sheet():
    try:
        credentials_dict = st.secrets["gcp_service_account"]
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(creds)
        
        # 1. 원본 시트 연결 (기존 파일)
        sheet_main = client.open("정산챗봇로그").sheet1
        
        # 2. 백업 시트 연결 (수정된 파일명 반영)
        try:
            # 여기가 수정되었습니다!
            sheet_backup = client.open("정산챗봇로그_백업용").sheet1
        except Exception as e:
            print(f"백업 시트 연결 실패: {e}")
            sheet_backup = None
            
        return sheet_main, sheet_backup # 두 개의 시트 객체를 반환
    except Exception as e:
        print(f"구글 시트 연결 오류: {e}")
        return None, None

def save_interaction(user_session_id, branch_name, question, answer):
    """
    저장 순서: [ID, 세션ID, 무역관, 질문, 답변, 피드백(공란), 의견(공란), 현지시간, 한국시간]
    """
    try:
        # 두 개의 시트를 받아옴
        sheet_main, sheet_backup = get_google_sheet()
        
        if sheet_main:
            unique_id = str(uuid.uuid4())
            
            # 1. 기준 시간(UTC) 구하기
            utc_now = datetime.now(pytz.utc)
            
            # 2. 한국 시간(KST) 변환
            kst_tz = pytz.timezone('Asia/Seoul')
            kst_time_str = utc_now.astimezone(kst_tz).strftime("%Y-%m-%d %H:%M:%S")
            
            # 3. 현지 시간(Local) 변환
            target_tz_name = BRANCH_TIMEZONES.get(branch_name, 'Asia/Seoul')
            try:
                local_tz = pytz.timezone(target_tz_name)
                local_time_str = utc_now.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
            except:
                local_time_str = kst_time_str

            row_data = [unique_id, user_session_id, branch_name, question, answer, "", "", local_time_str, kst_time_str]
            
            # [수정됨] 원본에 저장
            sheet_main.append_row(row_data)
            
            # [수정됨] 백업본에도 저장 (백업 시트가 연결되어 있을 때만 실행)
            if sheet_backup:
                try:
                    sheet_backup.append_row(row_data)
                except Exception as e:
                    print(f"백업 파일 저장 중 오류(무시됨): {e}")

            return unique_id
    except Exception as e:
        print(f"저장 실패: {e}")
    return None

def update_feedback(unique_id, feedback_score, feedback_text):
    """
    ID를 찾아 피드백과 의견을 업데이트 (양쪽 시트 모두 반영)
    """
    try:
        sheet_main, sheet_backup = get_google_sheet()
        
        # 이모지 -> 텍스트 변환 로직
        if feedback_score == "👍":
            final_score = "좋아요"
        elif feedback_score == "👎":
            final_score = "싫어요"
        else:
            final_score = feedback_score

        text_to_save = feedback_text if feedback_text else ""

        # 1. 원본 시트 업데이트 (필수)
        if sheet_main:
            try:
                cell = sheet_main.find(unique_id, in_column=1)
                if cell:
                    sheet_main.update_cell(cell.row, 6, final_score)
                    sheet_main.update_cell(cell.row, 7, text_to_save)
            except Exception as e:
                print(f"원본 업데이트 오류: {e}")
        
        # 2. 백업 시트 업데이트 (선택 - 실패해도 에러 안 나게 처리)
        if sheet_backup:
            try:
                cell_bk = sheet_backup.find(unique_id, in_column=1)
                if cell_bk:
                    sheet_backup.update_cell(cell_bk.row, 6, final_score)
                    sheet_backup.update_cell(cell_bk.row, 7, text_to_save)
            except Exception as e:
                # 백업 파일에서 ID를 못 찾거나 에러가 나도, 챗봇은 멈추면 안 됨
                print(f"백업 파일 업데이트 실패 (무시함): {e}")

        return True
    except Exception as e:
        print(f"업데이트 실패: {e}")
    return False

# --------------------------------------------------------------------------
# [기능 2] 로그인 (무역관 선택 + 비밀번호)
# --------------------------------------------------------------------------
def check_login():
    if "login_success" not in st.session_state:
        st.session_state["login_success"] = False
    if "user_branch" not in st.session_state:
        st.session_state["user_branch"] = ""

    if st.session_state["login_success"]:
        return True

    st.header("🔒 무역관 정산 챗봇 로그인")
    st.write("소속 무역관을 선택하고 비밀번호를 입력해주세요.")
    
    # [설정] 무역관 목록 (여기에 있는 이름이 위 BRANCH_TIMEZONES 키와 같아야 시간이 정확함)
    branch_options = ["선택해주세요"] + list(BRANCH_TIMEZONES.keys()) # 사전 키를 그대로 목록으로 사용
    
    selected_branch = st.selectbox("소속 무역관", branch_options)
    password_input = st.text_input("비밀번호", type="password")
    
    if st.button("접속하기") or password_input:
        if selected_branch == "선택해주세요":
            st.warning("⚠️ 소속 무역관을 선택해주세요!")
            return False
            
        if password_input == st.secrets["PASSWORD"]:
            st.session_state["login_success"] = True
            st.session_state["user_branch"] = selected_branch
            st.rerun()
        else:
            st.error("❌ 비밀번호가 틀렸습니다.")
            
    return False

if not check_login():
    st.stop()

# --------------------------------------------------------------------------
# [메인] 앱 UI 시작
# --------------------------------------------------------------------------
st.title("💰무역관 정산 챗봇")
st.caption(f"환영합니다! **{st.session_state.user_branch}** 담당자님 👋 서비스 개선을 위해 답변 하단의 [좋아요👍/싫어요👎] 선택 후 SUBMIT을 꼭 눌러주세요!")

load_dotenv()

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())

if "message_list" not in st.session_state:
    st.session_state.message_list = []

# --- 스타일 설정 ---
st.markdown(
    """
    <style>
        .main .block-container { padding-bottom: 120px !important; }
        [data-testid="stChatInput"] { bottom: 50px !important; background-color: transparent !important; }
        [data-testid="stChatInput"] > div { border-color: transparent !important; }
        .footer-disclaimer {
            position: fixed; left: 0; bottom: 0; width: 100%; height: 95px;
            background-color: #ffffff; color: #888888; text-align: center;
            font-size: 12px; display: flex; align-items: center; justify-content: center;
            z-index: 100; border-top: 1px solid #f0f0f0;
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

# --- 대화 기록 및 피드백 버튼 ---
for i, message in enumerate(st.session_state.message_list):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        if message["role"] == "ai":
            if message.get("source"):
                st.caption(message["source"])
            
            if message["content"] and message.get("row_id"):
                feedback_key = f"feedback_{i}" 
                feedback = streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="의견은 선택사항이나, 평가 저장을 위해 SUBMIT을 꼭 눌러주세요",
                    key=feedback_key,
                    align="flex-start"
                )

                if feedback:
                    score = feedback["score"]
                    text = feedback.get("text")
                    target_row_id = message["row_id"]
                    
                    if target_row_id:
                        update_feedback(target_row_id, score, text)
                        st.toast("피드백이 반영되었습니다!", icon="✅")

# --- 사용자 입력 처리 ---
if user_question := st.chat_input("해외무역관 정산에 대한 궁금한 내용을 물어보세요"):
    st.session_state.message_list.append({"role": "user", "content": user_question})
    st.session_state.message_list.append({"role": "ai", "content": "", "source": None, "row_id": None})
    st.rerun()

# --- 답변 생성 및 저장 ---
if (
    st.session_state.message_list
    and st.session_state.message_list[-1]["role"] == "ai"
    and st.session_state.message_list[-1]["content"] == ""
):
    user_question = st.session_state.message_list[-2]["content"]

    def _format_page_token(page_val):
        if page_val is None: return None
        if isinstance(page_val, int): return f"{page_val}p"
        s = str(page_val).strip()
        if not s: return None
        try:
            num = float(s)
            if num.is_integer(): return f"{int(num)}p"
        except ValueError: pass
        if re.fullmatch(r"\d+", s): return f"{s}p"
        return s

    def _dedup_preserve_order(items):
        seen = set()
        out = []
        for x in items:
            if x is None: continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _resolve_pdf_name(metadata: dict):
        name = (metadata or {}).get("origin_pdf") or (metadata or {}).get("source") or "알 수 없음"
        try: return Path(str(name)).name
        except Exception: return str(name)

    with st.spinner("답변을 생성하는 중입니다..."):
        ai_response_stream = get_ai_response(user_question)
        full_answer = ""
        source_info = None
        origins_order = []
        pages_by_origin = {}
        seen_pairs = set()

        for chunk in ai_response_stream:
            if "context" in chunk and source_info is None:
                docs = chunk["context"] or []
                for d in docs:  
                    md = getattr(d, "metadata", {}) or {}
                    origin = _resolve_pdf_name(md)
                    page_token = _format_page_token(md.get("page_num"))
                    if not page_token: continue
                    pair = (origin, page_token)
                    if pair in seen_pairs: continue
                    seen_pairs.add(pair)
                    if origin not in pages_by_origin:
                        pages_by_origin[origin] = []
                        origins_order.append(origin)
                    if page_token not in pages_by_origin[origin]:
                        pages_by_origin[origin].append(page_token)
                if origins_order:
                    parts = []
                    for origin in origins_order:
                        pages = _dedup_preserve_order(pages_by_origin.get(origin, []))
                        if not pages: continue
                        pages_str = ", ".join(pages)
                        parts.append(f"({origin}) {pages_str}")
                    if parts:
                        source_info = "📄출처: " + " / ".join(parts)
                    else:
                        source_info = "📄출처: 페이지 정보 없음"
                else:
                    source_info = "📄출처: 페이지 정보 없음"

            if "answer" in chunk:
                full_answer += chunk["answer"]

        # [저장] 시간 저장 부분 개선
        current_session_id = st.session_state.user_session_id
        current_branch = st.session_state.user_branch
        
        row_id = save_interaction(current_session_id, current_branch, user_question, full_answer)
        
        st.session_state.message_list[-1]["content"] = full_answer
        st.session_state.message_list[-1]["source"] = source_info
        st.session_state.message_list[-1]["row_id"] = row_id

        st.rerun()