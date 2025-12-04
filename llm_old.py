from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings 
from langchain_pinecone import PineconeVectorStore
#from langchain.chains import RetrievalQA 
#from langchain import hub 

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples
    
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = OpenAIEmbeddings(model = 'text-embedding-3-large') # 임베딩 모델을 large로 바꿔주기
    index_name = 'finance-new-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(
        search_type="similarity", # 점수순
        search_kwargs={"k": 3}    # k고정
        ) 
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
   
    # 어제 영화 봤어? > 인셉션 > 어땠어? --> 인셉션 어땠어?
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question " # 대화 기록과 사용자의 최신 질문이 주어졌을 때
        "which might reference context in the chat history, " # 질문이 대화 기록의 맥락을 참조할 수도 있으므로
        "formulate a standalone question which can be understood " # 대화 기록 없이도 이해될 수 있는 독립적인 질문으로 다시 작성하세요
        "without the chat history. Do NOT answer the question, " # 질문 자체가 이미 독립적이면 그대로 두고
        "just reformulate it if needed and otherwise return it as is." # 대답하지 말고 질문만 다시 제시하세요
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever
    
    
def get_llm(model = 'gpt-5.1'):
    llm = ChatOpenAI(model = model)
    return llm


def get_rag_chain(): # 챗봇의 엔진
    # 1. 기본 부품 준비
    llm = get_llm() # 언어 모델(GPT-4o)
    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{answer}"),
    ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 회사 재무팀에서 근무하는 해외무역관 정산 전문가입니다. 정산에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 자료에 없음이라고 답변해주세요"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever() # 대화 맥락을 이해하는 '스마트 검색기'

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )#.pick('answer')
    
    return conversational_rag_chain
    

def get_ai_response(user_message): # 1. 챗봇의 시작: 사용자 질문을 받음
    qa_chain = get_rag_chain()     # 2. 챗봇의 핵심 기능(RAG Chain)을 불러옵니다. 
    ai_response = qa_chain.stream( # 3. 사용자 메시지를 넣어 답변을 스트리밍 방식으로 받습니다.
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": "abc123"} # 4. 대화 세션을 지정합니다.
        },
    )
    return ai_response # 답변이 생성되는 대로 바로바로 UI에 전달