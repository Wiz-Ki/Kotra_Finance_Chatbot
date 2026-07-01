import re
import os
from pathlib import Path
import streamlit as st

from config import (
    get_google_sheet_backup_name,
    get_google_sheet_main_name,
    get_rag_max_source_lines,
)
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
st.set_page_config(page_title="해외무역관 AI 정산도우미", page_icon="💰")


def load_runtime_config():
    secret_paths = {
        "OPENAI_API_KEY": (("OPENAI_API_KEY",), ("OpenAI", "OPENAI_API_KEY")),
        "PINECONE_API_KEY": (("PINECONE_API_KEY",), ("Pinecone_API_Dev", "Pinecone_API_Dev")),
    }

    for key, paths in secret_paths.items():
        secret_value = None
        for path in paths:
            try:
                current = st.secrets
                for path_key in path:
                    current = current[path_key]
                secret_value = current
                break
            except Exception:
                continue
        if secret_value:
            os.environ[key] = str(secret_value)


load_runtime_config()


def local_login_bypass_enabled():
    local_secrets_path = APP_DIR / ".streamlit" / "secrets.toml"
    if not local_secrets_path.exists():
        return False
    try:
        return (
            st.secrets.get("APP_ENV") == "local"
            and bool(st.secrets.get("LOCAL_BYPASS_LOGIN", False))
        )
    except Exception:
        return False

# --------------------------------------------------------------------------
# [설정] 무역관별 시간대 매핑 (전체 무역관 리스트 반영)
# --------------------------------------------------------------------------
BRANCH_TIMEZONES = {
    # --- 공통 및 본사 ---
    "본사": "Asia/Seoul",
    "기타": "Asia/Seoul",
    "재무팀": "Asia/Seoul",
    "AI데이터팀": "Asia/Seoul",

    # --- 중국 (Asia/Shanghai) ---
    "베이징무역관": "Asia/Shanghai", "상하이무역관": "Asia/Shanghai", "광저우무역관": "Asia/Shanghai",
    "다롄무역관": "Asia/Shanghai", "청두무역관": "Asia/Shanghai", "칭다오무역관": "Asia/Shanghai",
    "우한무역관": "Asia/Shanghai", "샤먼무역관": "Asia/Shanghai", "난징무역관": "Asia/Shanghai",
    "선전무역관": "Asia/Shanghai", "항저우무역관": "Asia/Shanghai", "정저우무역관": "Asia/Shanghai",
    "창사무역관": "Asia/Shanghai", "시안무역관": "Asia/Shanghai", "텐진무역관": "Asia/Shanghai",
    "홍콩무역관": "Asia/Hong_Kong", "타이베이무역관": "Asia/Taipei", "선양무역관": "Asia/Shanghai",
    "충칭무역관": "Asia/Shanghai", "창춘무역관": "Asia/Shanghai", "하얼빈무역관": "Asia/Shanghai",

    # --- 유럽 (Europe/...) ---
    "프랑크푸르트무역관": "Europe/Berlin", "함부르크무역관": "Europe/Berlin", "뮌헨무역관": "Europe/Berlin",
    "런던무역관": "Europe/London", "파리무역관": "Europe/Paris", "밀라노무역관": "Europe/Rome",
    "마드리드무역관": "Europe/Madrid", "브뤼셀무역관": "Europe/Brussels", "암스테르담무역관": "Europe/Amsterdam",
    "취리히무역관": "Europe/Zurich", "스톡홀름무역관": "Europe/Stockholm", "코펜하겐무역관": "Europe/Copenhagen",
    "헬싱키무역관": "Europe/Helsinki", "빈무역관": "Europe/Vienna",
    "바르샤바무역관": "Europe/Warsaw", "부다페스트무역관": "Europe/Budapest", "프라하무역관": "Europe/Prague",
    "부쿠레슈티무역관": "Europe/Bucharest", "소피아무역관": "Europe/Sofia", "베오그라드무역관": "Europe/Belgrade",
    "자그레브무역관": "Europe/Zagreb", "브라티슬라바무역관": "Europe/Bratislava", "아테네무역관": "Europe/Athens",
    "리스본무역관": "Europe/Lisbon",

    # --- 북미 (America/...) ---
    "뉴욕무역관": "America/New_York", "워싱턴무역관": "America/New_York", "애틀랜타무역관": "America/New_York",
    "디트로이트무역관": "America/Detroit", "시카고무역관": "America/Chicago", "달라스무역관": "America/Chicago",
    "로스앤젤레스무역관": "America/Los_Angeles", "실리콘밸리무역관": "America/Los_Angeles",
    "토론토무역관": "America/Toronto", "밴쿠버무역관": "America/Vancouver",

    # --- 일본 (Asia/Tokyo) ---
    "도쿄무역관": "Asia/Tokyo", "오사카무역관": "Asia/Tokyo", "나고야무역관": "Asia/Tokyo", "후쿠오카무역관": "Asia/Tokyo",

    # --- 동남아시아/대양주 ---
    "방콕무역관": "Asia/Bangkok", "싱가포르무역관": "Asia/Singapore", "하노이무역관": "Asia/Ho_Chi_Minh",
    "호치민무역관": "Asia/Ho_Chi_Minh", "다낭무역관": "Asia/Ho_Chi_Minh", "자카르타무역관": "Asia/Jakarta",
    "수라바야무역관": "Asia/Jakarta", "쿠알라룸푸르무역관": "Asia/Kuala_Lumpur", "마닐라무역관": "Asia/Manila",
    "프놈펜무역관": "Asia/Phnom_Penh", "양곤무역관": "Asia/Yangon", "비엔티안무역관": "Asia/Vientiane",
    "시드니무역관": "Australia/Sydney", "멜버른무역관": "Australia/Melbourne", "오클랜드무역관": "Pacific/Auckland",

    # --- 서남아시아/중동/아프리카 ---
    "뉴델리무역관": "Asia/Kolkata", "뭄바이무역관": "Asia/Kolkata", "첸나이무역관": "Asia/Kolkata",
    "벵갈루루무역관": "Asia/Kolkata", "콜롬보무역관": "Asia/Colombo", "다카무역관": "Asia/Dhaka",
    "카라치무역관": "Asia/Karachi", "두바이무역관": "Asia/Dubai", "무스카트무역관": "Asia/Muscat",
    "리야드무역관": "Asia/Riyadh", "제다무역관": "Asia/Riyadh", "쿠웨이트무역관": "Asia/Kuwait",
    "도하무역관": "Asia/Doha", "암만무역관": "Asia/Amman", "베이루트무역관": "Asia/Beirut",
    "텔아비브무역관": "Asia/Tel_Aviv", "이스탄불무역관": "Europe/Istanbul", "카이로무역관": "Africa/Cairo",
    "카사블랑카무역관": "Africa/Casablanca", "알제무역관": "Africa/Algiers", "트리폴리무역관": "Africa/Tripoli",
    "나이로비무역관": "Africa/Nairobi", "아디스아바바무역관": "Africa/Addis_Ababa", "요하네스버그무역관": "Africa/Johannesburg",
    "라고스무역관": "Africa/Lagos", "아크라무역관": "Africa/Accra", "다르에스살람무역관": "Africa/Dar_es_Salaam",
    "암다바드무역관": "Asia/Kolkata", "콜카타무역관": "Asia/Kolkata", "테헤란무역관": "Asia/Tehran", "바그다드무역관": "Asia/Baghdad",
    "아비장무역관": "Africa/Abidjan", "마푸투무역관": "Africa/Maputo",

    # --- 중남미 ---
    "멕시코시티무역관": "America/Mexico_City", "파나마무역관": "America/Panama", "과테말라무역관": "America/Guatemala",
    "산토도밍고무역관": "America/Santo_Domingo", "보고타무역관": "America/Bogota", "리마무역관": "America/Lima",
    "상파울루무역관": "America/Sao_Paulo", "부에노스아이레스무역관": "America/Argentina/Buenos_Aires", "아바나무역관": "America/Havana",
    "산티아고무역관": "America/Santiago", "키토무역관": "America/Quito", "아순시온무역관": "America/Asuncion",

    # --- CIS (독립국가연합) ---
    "모스크바무역관": "Europe/Moscow", "블라디보스토크무역관": "Asia/Vladivostok", "노보시비르스크무역관": "Asia/Novosibirsk",
    "상트페테르부르크무역관": "Europe/Moscow", "알마티무역관": "Asia/Almaty", "타슈켄트무역관": "Asia/Tashkent", "트빌리시무역관": "Asia/Tbilisi",
    "키이우무역관": "Europe/Kyiv", "울란바토르무역관": "Asia/Ulaanbaatar", "바쿠무역관": "Asia/Baku", "민스크무역관": "Europe/Minsk"
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
        
        # 1. 원본 시트 연결
        sheet_main = client.open(get_google_sheet_main_name()).sheet1
        
        # 2. 백업 시트 연결 (수정된 파일명 반영)
        try:
            sheet_backup = client.open(get_google_sheet_backup_name()).sheet1
        except Exception as e:
            print(f"백업 시트 연결 실패: {e}")
            sheet_backup = None
            
        return sheet_main, sheet_backup # 두 개의 시트 객체를 반환
    except Exception as e:
        print(f"구글 시트 연결 오류: {e}")
        return None, None

def save_interaction(user_session_id, branch_name, question, answer, source_info=None): # [수정] source_info 인자 추가
    """
    저장 순서: [ID, 세션ID, 무역관, 질문, 답변(출처포함), 피드백(공란), 의견(공란), 현지시간, 한국시간]
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

            # [수정] 답변과 출처 합치기
            final_answer = answer
            if source_info:
                # 가독성을 위해 줄바꿈 두 번(\n\n) 후 출처 표기
                final_answer = f"{answer}\n\n{source_info}"

            # [수정] row_data에 answer 대신 final_answer 저장
            row_data = [unique_id, user_session_id, branch_name, question, final_answer, "", "", local_time_str, kst_time_str]
            
            # 원본에 저장
            sheet_main.append_row(row_data)
            
            # 백업본에도 저장 (백업 시트가 연결되어 있을 때만 실행)
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

    if local_login_bypass_enabled():
        st.session_state["login_success"] = True
        st.session_state["user_branch"] = str(st.secrets.get("LOCAL_BYPASS_BRANCH", "본사"))
        return True

    if st.session_state["login_success"]:
        return True

    st.header("🔒 AI 정산도우미 로그인")
    st.write("소속 무역관을 선택하고 비밀번호를 입력해주세요.")
    
    # --- [수정된 정렬 로직] ---
    # 1. 하단에 고정할 항목들 리스트
    fixed_items = ["본사", "기타", "재무팀", "AI데이터팀"]
    
    # 2. 전체 키 중에서 고정 항목을 제외한 '무역관'들만 추출하여 가나다 순으로 정렬
    trade_centers = sorted([k for k in BRANCH_TIMEZONES.keys() if k not in fixed_items])
    
    # 3. 최종 리스트: 선택해주세요 + 정렬된 무역관 + 고정 항목 순서로 합치기
    branch_options = ["선택해주세요"] + trade_centers + fixed_items
    # --------------------------
    
    selected_branch = st.selectbox("소속 무역관/본사", branch_options)
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
st.title("💰해외무역관 AI 정산도우미")
st.caption(f"환영합니다! **{st.session_state.user_branch}** 담당자님 👋 서비스 개선을 위해 답변 하단의 **[좋아요👍/싫어요👎]** 선택 후 SUBMIT을 꼭 눌러주세요!")

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())

if "message_list" not in st.session_state:
    st.session_state.message_list = []

# --- 스타일 설정 ---
st.markdown(
    """
    <style>
        /* 메인 영역 하단 여백: 푸터가 높아짐에 따라 컨텐츠가 가려지지 않게 조정 */
        .main .block-container { padding-bottom: 120px !important; }
        
        /* 채팅 입력창 위치: 푸터 위로 배치 (높이 조절) */
        [data-testid="stChatInput"] { 
            bottom: 40px !important; 
            background-color: transparent !important; 
        }
        [data-testid="stChatInput"] > div { border-color: transparent !important; }

        /* 커스텀 푸터 컨테이너 */
        .footer-container {
            position: fixed; 
            left: 0; 
            bottom: 0; 
            width: 100%; 
            height: 85px; /* 버튼을 포함하기 위해 높이 약간 증가 */
            background-color: #ffffff; 
            z-index: 100; 
            border-top: 1px solid #f0f0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start; /* 위쪽(라인)부터 배치 */
            padding-top: 10px; /* 라인과의 미세한 간격 */
        }

        /* 주의 문구 스타일 */
        .disclaimer-text {
            color: #888888; 
            font-size: 11px;
            text-align: center;
            margin-bottom: 10px; /* 문구와 버튼 사이 간격 */
            line-height: 1.4;
        }

        /* 푸터 바로가기 버튼 그룹 */
        .footer-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: calc(100% - 16px);
        }

        /* 푸터 바로가기 버튼 스타일 */
        .footer-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px clamp(8px, 2vw, 16px);
            background-color: #f8f9fa;
            color: #444444 !important;
            text-decoration: none !important;
            border: 1px solid #d1d1d1;
            border-radius: 6px;
            font-size: clamp(11px, 2.8vw, 13px);
            font-weight: 500;
            white-space: nowrap;
            transition: all 0.2s ease;
        }

        .footer-link:hover {
            background-color: #eeeeee;
            border-color: #bbbbbb;
            color: #000000 !important;
        }
    </style>

    <div class="footer-container">
        <div class="disclaimer-text">
            저는 아직 배우는 중이라 실수가 있을 수 있어요! 😅
            답변은 참고만 해주시고, 헷갈리는 부분은 꼭 재무팀 담당자분들께 확인 부탁드려요.
        </div>
        <div class="footer-links">
            <a href="https://kbei.org/whistle" target="_blank" rel="noopener noreferrer" class="footer-link">
                🚨 케이휘슬 익명신고 바로가기
            </a>
            <a href="https://moaform.com/q/HsV3Zj" target="_blank" rel="noopener noreferrer" class="footer-link">
                📊 만족도 조사 바로가기
            </a>
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
        # [추가] 답변 생성 중에 스피너 아래에 투명 공간을 만들어 스피너를 위로 밀어 올림
        loading_spacer = st.empty()
        with loading_spacer:
            st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
            
        ai_response_stream = get_ai_response(user_question, st.session_state.user_session_id)
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
                    source_lines = []
                    max_source_lines = get_rag_max_source_lines()
                    for origin in origins_order:
                        pages = _dedup_preserve_order(pages_by_origin.get(origin, []))
                        if not pages: continue
                        for page in pages:
                            source_lines.append(f"- ({origin}) {page}")
                            if len(source_lines) >= max_source_lines:
                                break
                        if len(source_lines) >= max_source_lines:
                            break
                    if source_lines:
                        source_info = "📄출처:\n\n" + "\n".join(source_lines)
                    else:
                        source_info = "📄출처: 페이지 정보 없음"
                else:
                    source_info = "📄출처: 페이지 정보 없음"

            if "answer" in chunk:
                full_answer += chunk["answer"]

        # [저장] 시간 저장 부분 개선
        current_session_id = st.session_state.user_session_id
        current_branch = st.session_state.user_branch
        
        # [수정] 함수 호출 시 source_info 인자 추가 전달
        row_id = save_interaction(current_session_id, current_branch, user_question, full_answer, source_info)
        
        st.session_state.message_list[-1]["content"] = full_answer
        st.session_state.message_list[-1]["source"] = source_info
        st.session_state.message_list[-1]["row_id"] = row_id

        st.rerun()
