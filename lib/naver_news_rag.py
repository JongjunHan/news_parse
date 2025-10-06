# -*- coding: utf-8 -*-
"""
네이버 뉴스 파싱 및 RAG 시스템 모듈
Python 3.11+ 호환
"""

import requests
import bs4
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


class NaverNewsParser:
    """네이버 뉴스를 검색하고 제목과 본문만 파싱하는 클래스"""

    def __init__(self, client_id: str, client_secret: str):
        """
        Args:
            client_id: 네이버 API Client ID
            client_secret: 네이버 API Client Secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.noise_texts = [
            '구독중 구독자 0 응원수 0 더보기',
            '쏠쏠정보 0 흥미진진 0 공감백배 0 분석탁월 0 후속강추 0',
            '댓글 본문 요약봇 본문 요약봇',
            '도움말 자동 추출 기술로 요약된 내용입니다. 요약 기술의 특성상 본문의 주요 내용이 제외될 수 있어, 전체 맥락을 이해하기 위해서는 기사 본문 전체보기를 권장합니다. 닫기',
            '텍스트 음성 변환 서비스 사용하기 성별 남성 여성 말하기 속도 느림 보통 빠름',
            '이동 통신망을 이용하여 음성을 재생하면 별도의 데이터 통화료가 부과될 수 있습니다. 본문듣기 시작',
            '닫기 글자 크기 변경하기 가1단계 작게 가2단계 보통 가3단계 크게 가4단계 아주크게 가5단계 최대크게 SNS 보내기 인쇄하기',
        ]

    def get_news_links(self, query: str, num_links: int = 10, sort: str = "date") -> list[str]:
        """
        네이버 뉴스 검색 API로 뉴스 링크 가져오기 (최신순)

        Args:
            query: 검색 쿼리
            num_links: 가져올 링크 수
            sort: 정렬 방식 ("date": 최신순, "sim": 유사도순) - 기본값: date (최신순)

        Returns:
            네이버 뉴스 링크 리스트 (최신 뉴스부터)
        """
        import urllib.parse

        # 쿼리 URL 인코딩
        encoded_query = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/news.json?query={encoded_query}&display={num_links}&sort={sort}"
        headers = {
            'X-Naver-Client-Id': self.client_id,
            'X-Naver-Client-Secret': self.client_secret
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)

            # HTTP 401 에러 상세 처리
            if response.status_code == 401:
                raise ValueError(
                    "네이버 API 인증 실패 (401 Unauthorized)\n"
                    "원인:\n"
                    "1. NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET가 잘못되었습니다.\n"
                    "2. .env 파일의 API 키에 공백이나 따옴표가 포함되어 있습니다.\n"
                    "3. 네이버 개발자 센터에서 애플리케이션이 비활성화되었습니다.\n\n"
                    "해결 방법:\n"
                    "1. https://developers.naver.com/apps/#/myapps 에서 API 키 확인\n"
                    "2. .env 파일에서 따옴표 제거 (예: NAVER_CLIENT_ID=abc123)\n"
                    "3. API 키 앞뒤 공백 제거\n"
                    "4. 애플리케이션에 '검색' API가 등록되어 있는지 확인"
                )

            response.raise_for_status()  # 기타 HTTP 에러 체크
            result = response.json()

            # API 응답에 'items' 키가 없는 경우 처리
            if 'items' not in result:
                # 에러 메시지 확인
                if 'errorMessage' in result:
                    error_msg = result['errorMessage']
                    error_code = result.get('errorCode', 'Unknown')
                    raise ValueError(f"네이버 API 오류 [{error_code}]: {error_msg}")
                else:
                    print(f"경고: API 응답에 'items'가 없습니다. 응답: {result}")
                    return []

            # items가 비어있는 경우
            if not result['items']:
                print(f"경고: '{query}' 검색 결과가 없습니다.")
                return []

            filtered_links = []
            for item in result['items']:
                link = item.get('link', '')
                if "n.news.naver.com/mnews/article/" in link or "n.news.naver.com/article/" in link:
                    filtered_links.append(link)

            return filtered_links

        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            raise ValueError(f"네이버 뉴스 API 요청 실패: {e}")
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            raise

    def parse_news_content(self, url: str) -> dict[str, str]:
        """
        뉴스 URL에서 제목과 본문만 추출

        Args:
            url: 뉴스 URL

        Returns:
            {'title': 제목, 'content': 본문, 'url': URL}
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # 제목 추출
            title_elem = soup.select_one('h2.media_end_head_headline, h1#title_area')
            title = title_elem.get_text(strip=True) if title_elem else ""

            # 본문 추출
            content_elem = soup.select_one('article#dic_area, div#articleBodyContents, div.newsct_article')
            if content_elem:
                # 스크립트, 스타일 태그 제거
                for script in content_elem(['script', 'style', 'figure', 'aside']):
                    script.decompose()
                content = content_elem.get_text(strip=True)
            else:
                content = ""

            return {
                'title': title,
                'content': content,
                'url': url
            }
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return {'title': '', 'content': '', 'url': url}

    def get_news_documents(self, links: list[str]) -> list[Document]:
        """
        뉴스 링크들로부터 제목과 본문을 파싱하여 Document 생성

        Args:
            links: 뉴스 URL 리스트

        Returns:
            Document 객체 리스트
        """
        documents = []
        for link in links:
            news_data = self.parse_news_content(link)
            if news_data['title'] and news_data['content']:
                # 제목과 본문을 합쳐서 저장
                full_text = f"제목: {news_data['title']}\n\n{news_data['content']}"
                doc = Document(
                    page_content=full_text,
                    metadata={
                        'source': news_data['url'],
                        'title': news_data['title']
                    }
                )
                documents.append(doc)
        return documents

    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        text = text.replace('\t', ' ').replace('\n', ' ')
        for _ in range(20):
            text = ' '.join(text.split())
        for noise in self.noise_texts:
            text = text.replace(noise, '')
        return text

    def preprocess(self, docs: list[Document]) -> list[Document]:
        """문서 전처리"""
        preprocessed_docs = []

        for doc in docs:
            content = doc.page_content

            # 불필요한 문구 제거
            try:
                content = content.split('구독 해지되었습니다.')[1]
            except:
                pass

            try:
                content = content.split('구독 메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!')[0]
            except:
                pass

            content = self.clean_text(content)
            doc.page_content = content
            preprocessed_docs.append(doc)

        return preprocessed_docs


class NewsVectorDB:
    """뉴스 문서를 Vector DB로 구축하는 클래스"""

    def __init__(self,
                 embedding_model_name: str = 'jhgan/ko-sroberta-nli',
                 use_openai: bool = False,
                 openai_model: str = 'text-embedding-3-small',
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200):
        """
        Args:
            embedding_model_name: HuggingFace 임베딩 모델 이름
            use_openai: OpenAI 임베딩 사용 여부
            openai_model: OpenAI 임베딩 모델 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        if use_openai:
            self.embedding_model = OpenAIEmbeddings(model=openai_model)
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                },
                show_progress=False,
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.db = None

    def build_from_documents(self,
                            docs: list[Document],
                            batch_size: int = 10,
                            persist_directory: str | None = None) -> Chroma:
        """
        문서로부터 Vector DB 구축

        Args:
            docs: Document 리스트
            batch_size: 배치 크기
            persist_directory: 영구 저장 디렉토리

        Returns:
            Chroma 벡터 DB 객체
        """
        # 문서 분할
        chunks = self.text_splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No chunks to process")

        # Vector DB 구축
        first_batch = chunks[:batch_size]
        kwargs = {
            'documents': first_batch,
            'embedding': self.embedding_model,
        }

        if persist_directory:
            kwargs['persist_directory'] = persist_directory

        self.db = Chroma.from_documents(**kwargs)

        # 나머지 배치 추가
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.db.add_documents(batch)

        return self.db

    def get_retriever(self, search_kwargs: dict | None = None):
        """
        Retriever 반환

        Args:
            search_kwargs: 검색 설정 (예: {'k': 4})

        Returns:
            VectorStore Retriever
        """
        if self.db:
            if search_kwargs:
                return self.db.as_retriever(search_kwargs=search_kwargs)
            return self.db.as_retriever()
        return None


class NewsRAG:
    """네이버 뉴스 RAG 시스템"""

    def __init__(self,
                 naver_client_id: str,
                 naver_client_secret: str,
                 openai_api_key: str | None = None,
                 use_openai_embedding: bool = False,
                 llm_model: str = "gpt-4o-mini",
                 temperature: float = 0.1):
        """
        Args:
            naver_client_id: 네이버 API Client ID
            naver_client_secret: 네이버 API Client Secret
            openai_api_key: OpenAI API Key (선택)
            use_openai_embedding: OpenAI 임베딩 사용 여부
            llm_model: LLM 모델 이름
            temperature: LLM temperature
        """
        self.parser = NaverNewsParser(naver_client_id, naver_client_secret)
        self.vector_db = NewsVectorDB(use_openai=use_openai_embedding)
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)

        # LLM 설정 저장
        self.llm_model = llm_model
        self.temperature = temperature

        # 세션 내 Vector DB 캐시 (메모리 기반)
        self.db_cache = {}
        self.current_query = None

        # 대화 히스토리 관리
        self.chat_history = []
        self.max_history = 10  # 최대 히스토리 개수

        # 챗봇 프롬프트 (대화형)
        self.system_prompt = '''당신은 네이버 뉴스를 기반으로 정보를 제공하는 친절한 AI 챗봇 어시스턴트입니다.

역할과 행동 지침:
1. 주어진 뉴스 Context를 바탕으로 사용자의 질문에 정확하고 친절하게 답변합니다.
2. 이전 대화 내용을 기억하고 맥락을 유지하며 대화합니다.
3. 답변은 자연스럽고 대화체로 작성하되, 전문적인 정보를 제공합니다.
4. Context에 정보가 없다면 솔직하게 "제공된 뉴스 자료에는 해당 정보가 없습니다"라고 답변합니다.
5. 필요시 출처 URL을 포함하여 신뢰성을 높입니다.
6. 3-5문장 정도로 간결하면서도 충분한 정보를 제공합니다.'''

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{chat_history}\n\n참고 자료:\n{context}\n\n질문: {question}")
        ])

    def format_docs(self, docs: list[Document]) -> str:
        """문서를 포맷팅"""
        return "\n\n---\n\n".join([
            f"{doc.page_content}\nURL: {doc.metadata['source']}"
            for doc in docs
        ])

    def format_chat_history(self) -> str:
        """대화 히스토리를 포맷팅"""
        if not self.chat_history:
            return "이전 대화 없음"

        formatted = "이전 대화:\n"
        for i, msg in enumerate(self.chat_history[-self.max_history:], 1):
            formatted += f"\n사용자: {msg['question']}\n어시스턴트: {msg['answer']}\n"
        return formatted

    def add_to_history(self, question: str, answer: str):
        """대화 히스토리에 추가"""
        self.chat_history.append({
            'question': question,
            'answer': answer
        })

        # 최대 개수 초과 시 오래된 것 삭제
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        print("대화 히스토리가 초기화되었습니다.")

    def show_history(self):
        """대화 히스토리 출력"""
        if not self.chat_history:
            print("\n대화 히스토리가 비어있습니다.")
            return

        print(f"\n대화 히스토리 ({len(self.chat_history)}개):")
        print("=" * 80)
        for i, msg in enumerate(self.chat_history, 1):
            print(f"\n[{i}] 사용자: {msg['question']}")
            print(f"    어시스턴트: {msg['answer'][:100]}...")
        print("=" * 80)

    def change_llm_model(self, model: str, temperature: float | None = None):
        """
        LLM 모델 변경

        Args:
            model: 새로운 LLM 모델 이름
            temperature: 새로운 temperature (선택, 기본값: 현재 설정 유지)
        """
        if temperature is None:
            temperature = self.temperature

        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.llm_model = model
        self.temperature = temperature
        print(f"LLM 모델이 {model}(으)로 변경되었습니다. (temperature: {temperature})")

    def build_vector_db(self,
                       query: str | list[str],
                       num_links: int = 10,
                       use_cache: bool = True,
                       sort: str = "date") -> Chroma:
        """
        검색 쿼리로 Vector DB 구축 (세션 내 메모리 캐시 사용)

        Args:
            query: 검색 쿼리 (문자열 또는 문자열 리스트)
            num_links: 각 쿼리당 검색할 뉴스 개수
            use_cache: 캐시된 DB 사용 여부
            sort: 정렬 방식 ("date": 최신순, "sim": 유사도순) - 기본값: date (최신순)

        Returns:
            Chroma 벡터 DB
        """
        # 쿼리를 리스트로 변환
        queries = [query] if isinstance(query, str) else query

        # 캐시 키 생성 (쿼리들을 정렬해서 합침)
        cache_key = " | ".join(sorted(queries))

        # 캐시에 이미 있으면 재사용
        if use_cache and cache_key in self.db_cache:
            print(f"[캐시] 세션 내 캐시된 Vector DB를 사용합니다: '{cache_key}'")
            self.vector_db.db = self.db_cache[cache_key]
            self.current_query = cache_key
            return self.vector_db.db

        # 새로운 DB 구축
        print(f"새로운 Vector DB를 구축합니다: '{cache_key}'")
        print(f"검색 주제 수: {len(queries)}개")
        print(f"정렬 방식: {'최신순' if sort == 'date' else '유사도순'}")

        # 모든 쿼리에 대해 뉴스 링크 가져오기 (최신순)
        all_links = []
        for q in queries:
            print(f"\n주제 '{q}' 검색 중 (최신 {num_links}개)...")
            links = self.parser.get_news_links(q, num_links, sort=sort)

            if not links:
                print(f"[경고] 검색 쿼리 '{q}'에 대한 뉴스를 찾을 수 없습니다.")
                continue

            print(f"[완료] '{q}': {len(links)}개의 뉴스 링크를 찾았습니다.")
            all_links.extend(links)

        if not all_links:
            raise ValueError(f"검색 쿼리에 대한 뉴스를 찾을 수 없습니다.")

        # 중복 링크 제거
        all_links = list(set(all_links))
        print(f"\n[완료] 총 {len(all_links)}개의 고유한 뉴스 링크를 찾았습니다.")

        # 뉴스 파싱 (제목과 본문만)
        print("\n뉴스 파싱 중...")
        docs = self.parser.get_news_documents(all_links)
        print(f"[완료] {len(docs)}개의 뉴스를 파싱했습니다.")

        # 전처리
        preprocessed_docs = self.parser.preprocess(docs)

        # Vector DB 구축 (메모리에만 저장, persist_directory 없음)
        print("\nVector DB 구축 중...")
        db = self.vector_db.build_from_documents(
            preprocessed_docs,
            persist_directory=None
        )

        # 캐시에 저장
        self.db_cache[cache_key] = db
        self.current_query = cache_key

        print(f"[완료] Vector DB 구축 완료! (메모리에 캐시됨)")

        return db

    def list_cached_queries(self) -> list[str]:
        """
        캐시된 검색 쿼리 목록 반환

        Returns:
            캐시된 쿼리 리스트
        """
        return list(self.db_cache.keys())

    def clear_cache(self, query: str | None = None):
        """
        캐시 삭제

        Args:
            query: 삭제할 특정 쿼리 (None이면 전체 삭제)
        """
        if query:
            if query in self.db_cache:
                del self.db_cache[query]
                print(f"'{query}' 캐시를 삭제했습니다.")
            else:
                print(f"'{query}' 캐시를 찾을 수 없습니다.")
        else:
            self.db_cache.clear()
            self.current_query = None
            print("모든 캐시를 삭제했습니다.")

    def chat(self, question: str) -> str:
        """
        대화형 챗봇 응답 (히스토리 포함)

        Args:
            question: 사용자 질문

        Returns:
            챗봇 응답
        """
        retriever = self.vector_db.get_retriever()
        if not retriever:
            raise ValueError("Vector DB가 구축되지 않았습니다. build_vector_db()를 먼저 실행하세요.")

        # 대화 히스토리 포맷팅
        chat_history_text = self.format_chat_history()

        # RAG 체인 구성 (대화 히스토리 포함)
        def prepare_input(query):
            docs = retriever.invoke(query)
            return {
                "chat_history": chat_history_text,
                "context": self.format_docs(docs),
                "question": query
            }

        rag_chain = (
            RunnablePassthrough()
            | prepare_input
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 답변 생성
        answer = rag_chain.invoke(question)

        # 히스토리에 추가
        self.add_to_history(question, answer)

        return answer

    def query(self, question: str, use_history: bool = True) -> str:
        """
        질문에 답변 (레거시 호환용)

        Args:
            question: 질문
            use_history: 대화 히스토리 사용 여부

        Returns:
            답변
        """
        if use_history:
            return self.chat(question)
        else:
            # 히스토리 없이 단순 질문-답변
            retriever = self.vector_db.get_retriever()
            if not retriever:
                raise ValueError("Vector DB가 구축되지 않았습니다.")

            docs = retriever.invoke(question)
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "참고 자료:\n{context}\n\n질문: {question}")
            ])

            chain = (
                RunnablePassthrough()
                | (lambda q: {"context": self.format_docs(docs), "question": q})
                | simple_prompt
                | self.llm
                | StrOutputParser()
            )

            return chain.invoke(question)

    def run(self, search_query: str, question: str, num_links: int = 10, use_cache: bool = True) -> str:
        """
        검색부터 답변까지 전체 파이프라인 실행

        Args:
            search_query: 네이버 뉴스 검색 쿼리
            question: 질문
            num_links: 검색할 뉴스 개수
            use_cache: 캐시된 DB 사용 여부

        Returns:
            답변
        """
        # Vector DB 구축 (캐시된 DB가 있으면 재사용)
        self.build_vector_db(search_query, num_links, use_cache=use_cache)

        # 질문 답변
        return self.query(question)


def main():
    """대화형 인터페이스로 뉴스 RAG 시스템 실행"""
    import os
    from dotenv import load_dotenv

    print("=" * 80)
    print("네이버 뉴스 RAG 시스템")
    print("=" * 80)

    # .env 파일에서 환경변수 로드
    load_dotenv()

    # API 키 설정 (.env 파일에서 먼저 확인, 없으면 입력받기)
    print("\n[API 키 설정]")

    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # .env에서 로드되었는지 확인
    if naver_client_id and naver_client_secret and openai_api_key:
        print("[완료] .env 파일에서 API 키를 불러왔습니다.")
        print(f"  - NAVER_CLIENT_ID: {naver_client_id[:10]}...")
        print(f"  - NAVER_CLIENT_SECRET: {naver_client_secret[:5]}...")
        print(f"  - OPENAI_API_KEY: {openai_api_key[:10]}...")
    else:
        print(".env 파일이 없거나 API 키가 설정되지 않았습니다.")
        print("API 키를 직접 입력해주세요.\n")

        if not naver_client_id:
            naver_client_id = input("NAVER_CLIENT_ID를 입력하세요: ").strip()
        if not naver_client_secret:
            naver_client_secret = input("NAVER_CLIENT_SECRET를 입력하세요: ").strip()
        if not openai_api_key:
            openai_api_key = input("OPENAI_API_KEY를 입력하세요: ").strip()

    if not naver_client_id or not naver_client_secret:
        print("네이버 API 키가 입력되지 않았습니다.")
        return

    if not openai_api_key:
        print("OpenAI API 키가 입력되지 않았습니다.")
        return

    # 환경변수에 OpenAI API 키 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # LLM 모델 선택
    print("\n[LLM 모델 선택]")
    print("1. gpt-4o-mini (빠르고 저렴)")
    print("2. gpt-4o (높은 품질)")
    print("3. gpt-3.5-turbo (가장 저렴)")
    print("4. gpt-4-turbo (균형잡힌 성능)")

    model_choice = input("\n모델을 선택하세요 (1-4, 기본값: 1): ").strip()

    model_map = {
        "1": "gpt-4o-mini",
        "2": "gpt-4o",
        "3": "gpt-3.5-turbo",
        "4": "gpt-4-turbo",
        "": "gpt-4o-mini"
    }

    llm_model = model_map.get(model_choice, "gpt-4o-mini")
    print(f"선택된 모델: {llm_model}")

    # 임베딩 모델 선택
    print("\n[임베딩 모델 선택]")
    print("1. HuggingFace (무료, 한국어 특화)")
    print("2. OpenAI (유료, 높은 품질)")

    embedding_choice = input("\n임베딩 모델을 선택하세요 (1-2, 기본값: 1): ").strip()
    use_openai_embedding = embedding_choice == "2"

    embedding_name = "OpenAI" if use_openai_embedding else "HuggingFace (jhgan/ko-sroberta-nli)"
    print(f"선택된 임베딩 모델: {embedding_name}")

    # NewsRAG 시스템 초기화
    try:
        print("\nRAG 시스템 초기화 중...")
        rag = NewsRAG(
            naver_client_id=naver_client_id,
            naver_client_secret=naver_client_secret,
            openai_api_key=openai_api_key,
            use_openai_embedding=use_openai_embedding,
            llm_model=llm_model,
            temperature=0.1
        )
        print("RAG 시스템 초기화 완료!")
    except Exception as e:
        print(f"RAG 시스템 초기화 실패: {e}")
        return

    # 뉴스 주제 입력 및 파싱
    print("\n" + "=" * 80)
    print("[뉴스 검색 및 파싱]")
    print("=" * 80)

    search_query_input = input("\n검색할 뉴스 주제를 입력하세요 (여러 주제는 쉼표로 구분, 예: AI 기술, LLM, 생성형 AI): ").strip()

    if not search_query_input:
        print("검색 주제가 입력되지 않았습니다.")
        return

    # 쉼표로 구분된 여러 주제를 배열로 변환
    queries = [q.strip() for q in search_query_input.split(',') if q.strip()]

    if not queries:
        print("검색 주제가 입력되지 않았습니다.")
        return

    # 단일 쿼리면 문자열, 여러 개면 리스트로 사용
    search_query = queries[0] if len(queries) == 1 else queries

    # 뉴스 개수 선택
    print("\n[검색할 뉴스 개수 선택]")
    print("1. 5개 (빠름)")
    print("2. 10개 (권장)")
    print("3. 20개 (보통)")
    print("4. 30개 (느림)")
    print("5. 직접 입력")

    num_choice = input("\n선택하세요 (1-5, 기본값: 2): ").strip()

    num_map = {
        "1": 5,
        "2": 10,
        "3": 20,
        "4": 30,
        "": 10
    }

    if num_choice == "5":
        custom_num = input("검색할 뉴스 개수를 입력하세요: ").strip()
        num_links = int(custom_num) if custom_num.isdigit() and int(custom_num) > 0 else 10
    else:
        num_links = num_map.get(num_choice, 10)

    print(f"선택된 뉴스 개수: {num_links}개")

    # 캐시 키 생성
    if isinstance(search_query, str):
        cache_key = search_query
        display_query = search_query
    else:
        cache_key = " | ".join(sorted(search_query))
        display_query = ", ".join(search_query)

    # 캐시 확인 및 사용 여부 선택
    use_cache = True

    if cache_key in rag.list_cached_queries():
        print(f"\n[캐시] 세션 내 캐시된 Vector DB를 발견했습니다: '{cache_key}'")
        use_choice = input("캐시된 DB를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
        use_cache = use_choice != 'n'

        if not use_cache:
            print("새로운 DB를 구축합니다.")
    else:
        print(f"\n'{display_query}' 주제로 {num_links}개의 뉴스를 검색하고 Vector DB를 구축합니다...")

    try:
        rag.build_vector_db(search_query, num_links, use_cache=use_cache)
    except Exception as e:
        print(f"Vector DB 구축 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 챗봇 대화 루프
    print("\n" + "=" * 80)
    print("[네이버 뉴스 챗봇]")
    print("=" * 80)
    print("안녕하세요! 네이버 뉴스 기반 AI 챗봇입니다.")
    print(f"현재 주제: '{display_query}'")
    print("\n사용 가능한 명령어:")
    print("  - 질문 입력: 뉴스 정보를 바탕으로 답변합니다")
    print("  - 'new': 새로운 주제로 검색")
    print("  - 'cache': 캐시된 주제 목록 보기")
    print("  - 'history': 대화 히스토리 보기")
    print("  - 'clear': 대화 히스토리 초기화")
    print("  - 'quit', 'exit', 'q': 종료")

    while True:
        print("\n" + "-" * 80)
        question = input("\n질문: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("프로그램을 종료합니다.")
            break

        if question.lower() == 'new':
            # 새로운 주제로 검색
            search_query_input = input("\n검색할 뉴스 주제를 입력하세요 (여러 주제는 쉼표로 구분): ").strip()
            if not search_query_input:
                print("검색 주제가 입력되지 않았습니다.")
                continue

            # 쉼표로 구분된 여러 주제를 배열로 변환
            queries = [q.strip() for q in search_query_input.split(',') if q.strip()]

            if not queries:
                print("검색 주제가 입력되지 않았습니다.")
                continue

            # 단일 쿼리면 문자열, 여러 개면 리스트로 사용
            search_query = queries[0] if len(queries) == 1 else queries

            # 뉴스 개수 선택
            print("\n[검색할 뉴스 개수 선택]")
            print("1. 5개 (빠름)")
            print("2. 10개 (권장)")
            print("3. 20개 (보통)")
            print("4. 30개 (느림)")
            print("5. 직접 입력")

            num_choice = input("\n선택하세요 (1-5, 기본값: 2): ").strip()

            num_map = {
                "1": 5,
                "2": 10,
                "3": 20,
                "4": 30,
                "": 10
            }

            if num_choice == "5":
                custom_num = input("검색할 뉴스 개수를 입력하세요: ").strip()
                num_links = int(custom_num) if custom_num.isdigit() and int(custom_num) > 0 else 10
            else:
                num_links = num_map.get(num_choice, 10)

            print(f"선택된 뉴스 개수: {num_links}개")

            # 캐시 키 생성
            if isinstance(search_query, str):
                cache_key = search_query
                display_query = search_query
            else:
                cache_key = " | ".join(sorted(search_query))
                display_query = ", ".join(search_query)

            # 캐시 확인 및 사용 여부 선택
            use_cache = True

            if cache_key in rag.list_cached_queries():
                print(f"\n[캐시] 세션 내 캐시된 Vector DB를 발견했습니다: '{cache_key}'")
                use_choice = input("캐시된 DB를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
                use_cache = use_choice != 'n'

                if not use_cache:
                    print("새로운 DB를 구축합니다.")
            else:
                print(f"\n'{display_query}' 주제로 {num_links}개의 뉴스를 검색하고 Vector DB를 구축합니다...")

            try:
                rag.build_vector_db(search_query, num_links, use_cache=use_cache)
                print(f"\n[완료] 현재 주제가 '{display_query}'(으)로 변경되었습니다.")

                # 대화 히스토리 초기화 여부 확인
                if rag.chat_history:
                    clear_choice = input("새로운 주제로 변경 시 대화 히스토리를 초기화하시겠습니까? (y/n, 기본값: y): ").strip().lower()
                    if clear_choice != 'n':
                        rag.clear_history()
            except Exception as e:
                print(f"Vector DB 구축 실패: {e}")
            continue

        if question.lower() == 'cache':
            # 캐시된 주제 목록 표시
            cached_queries = rag.list_cached_queries()
            if cached_queries:
                print(f"\n캐시된 주제 목록 ({len(cached_queries)}개):")
                for i, query in enumerate(cached_queries, 1):
                    marker = "*" if query == rag.current_query else " "
                    print(f"{marker} {i}. {query}")
                print("\n* = 현재 사용 중인 주제")
            else:
                print("\n캐시된 주제가 없습니다.")
            continue

        if question.lower() == 'history':
            # 대화 히스토리 표시
            rag.show_history()
            continue

        if question.lower() == 'clear':
            # 대화 히스토리 초기화
            rag.clear_history()
            continue

        if not question:
            print("질문을 입력해주세요.")
            continue

        # 챗봇 답변 생성 (히스토리 포함)
        print("\n[처리중] 챗봇이 답변을 생성하고 있습니다...")
        try:
            answer = rag.chat(question)
            print(f"\n[챗봇 답변]\n{answer}")
        except Exception as e:
            print(f"\n[오류] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
