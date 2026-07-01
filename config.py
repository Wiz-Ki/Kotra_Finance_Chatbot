PINECONE_INDEX_NAME = "kotra-finance-chatbot"
PINECONE_NAMESPACES = ["finance", "esg"]

GOOGLE_SHEET_MAIN_NAME = "정산챗봇로그_dev"
GOOGLE_SHEET_BACKUP_NAME = "정산챗봇로그_백업용_dev"

RAG_ROUTE_K = 8
RAG_BOTH_K = 6
RAG_FINAL_K = 4
RAG_SCORE_THRESHOLD = 0.32
RAG_MAX_SOURCE_LINES = 4

CHAT_HISTORY_SESSION_TTL_SECONDS = 3 * 60 * 60
CHAT_HISTORY_MAX_TURNS = 10

FINANCE_NAMESPACES = ["finance", "cal_guide", "edu_material"]

ESG_NAMESPACES = [
    "esg",
    "public_official_conflict_interest_act",
    "improper_solicitation_graft_act",
    "workplace_human_rights_guideline",
    "kotra_conflict_interest_guideline",
    "kotra_code_of_conduct",
]

ROUTER_SYSTEM_PROMPT = """
# 작업
대화 기록과 사용자의 최신 질문을 보고 RAG 검색 계획을 JSON으로 작성하세요.

# 출력 JSON
반드시 아래 키만 포함하는 JSON 객체 하나만 출력하세요.
{{
  "route": "finance" | "esg" | "both",
  "rewritten_query": "검색에 사용할 독립적인 한국어 질문",
  "confidence": 0.0
}}

# 라우팅 기준
- finance: 재무 정산, 전도자금, 공용카드, ERP, 계좌, 장부, 영수증, 출장비, 식비, 비목, 운영비, 전표, 자금 집행.
- esg: 윤리, 준법, 이해충돌, 청탁금지, 금품/선물 수수, 외부강의, 행동강령, 괴롭힘, 인권침해, 신고/회피/반환 절차.
- both: 재무 처리와 윤리·준법 판단이 모두 필요하거나, 어느 쪽인지 애매한 질문.

# 재작성 규칙
- 사용자의 핵심 명사, 조건, 문서명, 조항 번호, 금액, 기한을 유지하세요.
- 질문에 없는 제도명, 조건, 금액, 절차를 추측해서 추가하지 마세요.
- 대명사나 생략 표현은 대화 기록에서 확인되는 범위에서만 보완하세요.
- 애매하거나 복합적이면 route는 반드시 "both"로 선택하세요.
- confidence는 라우팅 판단에 대한 내부 참고값입니다. 0.0부터 1.0까지 숫자로 출력하세요.
- JSON 외의 설명, 마크다운, 코드블록을 출력하지 마세요.
""".strip()


def get_google_sheet_main_name() -> str:
    return GOOGLE_SHEET_MAIN_NAME


def get_google_sheet_backup_name() -> str:
    return GOOGLE_SHEET_BACKUP_NAME


def get_pinecone_index_name() -> str:
    return PINECONE_INDEX_NAME


def get_pinecone_namespaces() -> list[str]:
    return PINECONE_NAMESPACES


def get_rag_route_k() -> int:
    return RAG_ROUTE_K


def get_rag_both_k() -> int:
    return RAG_BOTH_K


def get_rag_final_k() -> int:
    return RAG_FINAL_K


def get_rag_score_threshold() -> float:
    return RAG_SCORE_THRESHOLD


def get_rag_max_source_lines() -> int:
    return max(1, RAG_MAX_SOURCE_LINES)


def get_chat_history_session_ttl_seconds() -> int:
    return max(60, CHAT_HISTORY_SESSION_TTL_SECONDS)


def get_chat_history_max_turns() -> int:
    return max(1, CHAT_HISTORY_MAX_TURNS)


answer_examples = [
    {
        "input": "공용카드를 추가로 발급받으려면 어떻게 해야 해?",
        "evidence": (
            "공용카드 추가 발급은 재무팀의 사전 승인 후 공문으로 진행한다. "
            "발급 후 ERP의 조직망공용카드관리 메뉴에 등록한다."
        ),
        "answer": (
            "재무팀의 사전 승인을 받은 뒤 공문으로 추가 발급을 진행해야 합니다. "
            "발급된 카드는 ERP의 `전도관리 – 조직망 계좌/카드정보 – "
            "조직망공용카드관리`에 등록하세요."
        ),
    },
    {
        "input": "직무관련자가 사적이해관계자인 것을 알게 됐어. 어떻게 해야 해?",
        "evidence": (
            "[법률] 사적이해관계자임을 안 날부터 14일 이내에 소속기관장에게 신고하고 회피를 신청한다.\n"
            "[내부지침] 별지 제1호 서식을 이해충돌방지담당관에게 서면, 전자문서 또는 "
            "청렴포털 표준신고시스템으로 제출한다."
        ),
        "answer": (
            "알게 된 날부터 14일 이내에 소속기관장에게 신고하고 회피를 신청해야 합니다. "
            "KOTRA에서는 별지 제1호 서식을 작성해 이해충돌방지담당관에게 서면, 전자문서 "
            "또는 청렴포털 표준신고시스템으로 제출하세요."
        ),
    },
    {
        "input": "외부강의 사례금을 받아도 돼?",
        "evidence": (
            "외부강의 사례금은 대통령령이 정한 금액을 초과할 수 없다. 사례금을 받는 외부강의는 마친 날부터 "
            "10일 이내에 신고하되, 요청자가 국가나 지방자치단체인 경우는 예외다."
        ),
        "answer": (
            "사례금은 정해진 상한을 초과할 수 없고, 사례금을 받는 외부강의는 원칙적으로 "
            "마친 날부터 10일 이내에 신고해야 합니다. 다만 국가나 지방자치단체가 요청한 경우는 "
            "신고 예외입니다. 요청한 기관이 국가 또는 지방자치단체인지 알려주세요."
        ),
    },
    {
        "input": "해외출장지에서 갈 만한 식당을 추천해줘.",
        "evidence": "검색 결과 없음",
        "answer": (
            "제공된 자료에서 확인할 수 없습니다. "
            "출장 식비의 정산 기준을 확인하려는 것인지 알려주세요."
        ),
    },
]
