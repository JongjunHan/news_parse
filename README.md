# 네이버 뉴스 RAG 챗봇

네이버 뉴스 API를 활용한 RAG(Retrieval-Augmented Generation) 기반 대화형 AI 챗봇입니다.

## 주요 기능

- **최신 뉴스 자동 검색**: 네이버 뉴스에서 최신순으로 뉴스를 검색하여 항상 최신 정보 제공
- **여러 주제 동시 검색**: 쉼표로 구분하여 여러 뉴스 주제를 한 번에 검색 가능
- **정렬 방식 선택**: 최신순 또는 유사도순 정렬 선택 가능 (기본값: 최신순)
- **Vector DB 구축**: 뉴스 제목과 본문만 파싱하여 검색 가능한 벡터 데이터베이스로 구축
- **대화형 챗봇**: 뉴스 내용을 기반으로 자연스러운 대화 제공
- **세션 메모리 캐시**: 검색한 주제들을 메모리에 캐싱하여 빠른 전환 가능
- **대화 히스토리 관리**: 이전 대화 내용을 기억하며 맥락 기반 응답 제공
- **다크모드 WebUI**: 심플하고 직관적인 다크 테마 웹 인터페이스

## 기술 스택

- **Python 3.11+**: 최신 타입 힌팅 지원
- **LangChain**: RAG 파이프라인 구축
- **OpenAI GPT**: LLM 모델 (gpt-4o-mini, gpt-4o 등)
- **ChromaDB**: 벡터 데이터베이스
- **HuggingFace**: 한국어 임베딩 모델 (jhgan/ko-sroberta-nli)
- **Flask**: WebUI 서버
- **BeautifulSoup**: 네이버 뉴스 파싱

## 설치 방법

### 1. 저장소 클론 또는 다운로드

```bash
cd news_parse
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

## 환경 변수 설정

### 1. .env 파일 생성

프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
cp .env.example .env
```

### 2. API 키 입력

`.env` 파일을 열어서 API 키를 입력하세요 (**따옴표 없이!**):

```env
NAVER_CLIENT_ID=abc123xyz
NAVER_CLIENT_SECRET=ABC123XYZ
OPENAI_API_KEY=sk-abc123xyz
```

**주의:** 따옴표를 사용하지 마세요!
- ❌ 잘못된 예: `NAVER_CLIENT_ID="abc123xyz"`
- ✅ 올바른 예: `NAVER_CLIENT_ID=abc123xyz`

### 3. 환경 변수 검증 (권장)

설정이 올바른지 확인하세요:

```bash
python check_env.py
```

이 스크립트는 다음을 확인합니다:
- .env 파일이 존재하는지
- API 키가 설정되었는지
- 따옴표가 포함되지 않았는지
- API 키 길이가 적절한지

### API 키 발급 방법

**네이버 API**
1. [네이버 개발자 센터](https://developers.naver.com/main/) 접속
2. 애플리케이션 등록
3. 검색 API 선택
4. Client ID, Client Secret 발급

**OpenAI API**
1. [OpenAI Platform](https://platform.openai.com/) 접속
2. API Keys 메뉴에서 새 키 생성

## 빠른 시작

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. .env 파일 생성 및 API 키 입력
cp .env.example .env
nano .env  # API 키 입력

# 3. 환경 변수 검증 (권장)
python check_env.py

# 4. WebUI 실행
python webui.py
```

## 사용 방법

### WebUI 실행 (권장)

```bash
python webui.py
```

- 서버 시작 시 브라우저가 자동으로 열립니다
- URL: http://localhost:5000
- 세션 유지 기간: 30일
- 콘솔에서 API 키 확인 메시지 표시

### CLI 실행

```bash
python -m lib.naver_news_rag
```

또는

```bash
cd lib
python naver_news_rag.py
```

## User Flow

### 1. check_env.py (환경 변수 검증)

```
[시작]
   ↓
[.env 파일 존재 확인]
   ↓
   ├─ 파일 없음 → [오류 메시지 + 해결 방법 안내] → [종료]
   ↓
   └─ 파일 있음
      ↓
   [.env 파일 로드]
      ↓
   [API 키 검증]
      ├─ NAVER_CLIENT_ID 확인
      │  ├─ 미설정 → [오류]
      │  ├─ 따옴표 포함 → [경고]
      │  └─ 정상 → [길이 표시]
      │
      ├─ NAVER_CLIENT_SECRET 확인
      │  ├─ 미설정 → [오류]
      │  ├─ 따옴표 포함 → [경고]
      │  └─ 정상 → [길이 표시]
      │
      └─ OPENAI_API_KEY 확인
         ├─ 미설정 → [오류]
         ├─ 따옴표 포함 → [경고]
         └─ 정상 → [길이 표시]
   ↓
[결과 출력]
   ├─ 모두 정상 → [✅ 성공 메시지 + 실행 가이드]
   └─ 오류 있음 → [❌ 오류 메시지 + 해결 방법]
   ↓
[종료]
```

**주요 기능:**
- .env 파일 존재 여부 확인
- API 키 형식 검증 (따옴표, 공백 체크)
- API 키 길이 확인
- 상세한 오류 메시지 및 해결 방법 제공

---

### 2. webui.py (Flask 웹 서버)

```
[시작]
   ↓
[Flask 앱 초기화]
   ├─ 세션 암호화 키 생성
   └─ 세션 설정 (30일 유지)
   ↓
[서버 시작]
   ├─ 포트 5000 바인딩
   ├─ 브라우저 자동 실행 (1.5초 후)
   └─ 서버 정보 출력
   ↓
[사용자 접속] → http://localhost:5000
   ↓
[세션 관리]
   ├─ 세션 ID 확인
   │  ├─ 없음 → [새 세션 ID 생성]
   │  └─ 있음 → [기존 세션 사용]
   ↓
   └─ [RAG 인스턴스 확인]
      ├─ 없음 → [새 RAG 인스턴스 생성]
      │         ├─ .env에서 API 키 로드
      │         ├─ API 키 검증
      │         └─ NewsRAG 객체 생성
      └─ 있음 → [기존 인스턴스 재사용]
   ↓
[사용자 인터랙션]
   │
   ├─ [주제 검색] → POST /api/build_db
   │     ├─ 검색 쿼리 입력 (단일 또는 쉼표로 구분)
   │     ├─ 정렬 방식 선택 (최신순/유사도순)
   │     ├─ 뉴스 개수 설정 (5~30개)
   │     ↓
   │     ├─ 캐시 확인
   │     │  ├─ 캐시 있음 → [캐시된 Vector DB 사용]
   │     │  └─ 캐시 없음 → [Vector DB 구축]
   │     │                  ├─ 네이버 API 호출
   │     │                  ├─ 뉴스 파싱
   │     │                  ├─ 임베딩 생성
   │     │                  └─ ChromaDB 저장
   │     ↓
   │     └─ [성공 응답] → 캐시 상태 및 현재 주제 표시
   │
   ├─ [챗봇 대화] → POST /api/chat
   │     ├─ 질문 입력
   │     ├─ Vector DB 존재 확인
   │     │  ├─ 없음 → [오류: 먼저 주제 검색 필요]
   │     │  └─ 있음 → [RAG 파이프라인 실행]
   │     │             ├─ 유사 문서 검색
   │     │             ├─ 대화 히스토리 결합
   │     │             ├─ LLM 답변 생성
   │     │             └─ 히스토리 저장
   │     ↓
   │     └─ [답변 반환] → 대화 횟수 업데이트
   │
   ├─ [대화 히스토리 조회] → GET /api/history
   │     └─ [히스토리 목록 반환]
   │
   ├─ [대화 히스토리 초기화] → POST /api/history/clear
   │     └─ [히스토리 삭제] → [성공 메시지]
   │
   ├─ [캐시 목록 조회] → GET /api/cache
   │     └─ [캐시된 주제 목록 + 현재 주제 반환]
   │
   ├─ [캐시 삭제] → POST /api/cache/clear
   │     ├─ 특정 주제 지정 → [해당 주제만 삭제]
   │     └─ 주제 미지정 → [모든 캐시 삭제]
   │
   ├─ [상태 확인] → GET /api/status
   │     └─ [현재 주제, 캐시 목록, 대화 횟수, DB 상태 반환]
   │
   └─ [LLM 모델 변경] → POST /api/change_model
         ├─ 모델 선택 (gpt-4o-mini, gpt-4o 등)
         ├─ 모델 검증
         └─ [모델 변경] → [성공 메시지]
   ↓
[세션 유지] (30일간 세션별 RAG 인스턴스 격리)
   ↓
[종료] (Ctrl+C)
```

**주요 특징:**
- 세션별 독립적인 RAG 인스턴스 관리
- 자동 브라우저 실행
- Vector DB 캐싱으로 빠른 주제 전환
- REST API 기반 비동기 통신
- 다크모드 WebUI

---

### 3. lib/naver_news_rag.py (RAG 시스템 / CLI)

#### CLI 모드 실행 시:
```
[시작] python -m lib.naver_news_rag
   ↓
[API 키 로드] (.env 파일)
   ↓
[NewsRAG 인스턴스 생성]
   ↓
[무한 루프 - 사용자 인터랙션]
   │
   ├─ [검색 쿼리 입력]
   │     ├─ 쉼표로 구분된 여러 주제 입력
   │     ↓
   │     [Vector DB 구축]
   │        ├─ NaverNewsParser 초기화
   │        ├─ 각 쿼리마다 뉴스 검색
   │        │  ├─ 네이버 API 호출 (최신순)
   │        │  ├─ HTML 파싱 (BeautifulSoup)
   │        │  ├─ 제목 + 본문 추출
   │        │  └─ Document 생성
   │        ↓
   │        ├─ NewsVectorDB 구축
   │        │  ├─ RecursiveTextSplitter (청킹)
   │        │  ├─ HuggingFace 임베딩 생성
   │        │  └─ ChromaDB에 벡터 저장
   │        ↓
   │        └─ 세션 캐시에 저장
   │
   ├─ [대화 루프]
   │     ├─ 사용자 질문 입력
   │     ↓
   │     ├─ [특수 명령어 처리]
   │     │  ├─ "새로운 검색" → Vector DB 구축으로 이동
   │     │  ├─ "히스토리" → 대화 히스토리 출력
   │     │  ├─ "캐시 목록" → 캐시된 주제 목록 출력
   │     │  └─ "종료" → 프로그램 종료
   │     ↓
   │     └─ [RAG 체인 실행]
   │        ├─ ChromaDB에서 유사 문서 검색
   │        ├─ 대화 히스토리 결합 (최대 10개)
   │        ├─ ChatPromptTemplate 구성
   │        ├─ OpenAI LLM 호출
   │        └─ 답변 생성 → 히스토리 저장
   │     ↓
   │     └─ [답변 출력] → 반복
   │
   └─ [루프 계속]
```

#### 프로그래밍 방식 사용:
```
[라이브러리 임포트]
   ↓
[NewsRAG 인스턴스 생성]
   ├─ API 키 설정
   ├─ LLM 모델 선택
   └─ 임베딩 모델 선택
   ↓
[Vector DB 구축]
   ├─ build_vector_db(query, num_links, sort)
   ├─ 캐시 관리
   └─ 성공/실패 반환
   ↓
[대화]
   ├─ chat(question)
   ├─ 답변 생성
   └─ 히스토리 자동 저장
   ↓
[유틸리티]
   ├─ show_history() - 대화 기록 출력
   ├─ clear_history() - 히스토리 초기화
   ├─ list_cached_queries() - 캐시 목록
   └─ change_llm_model() - 모델 변경
```

**주요 클래스:**
- **NaverNewsParser**: 네이버 API 검색 + HTML 파싱
- **NewsVectorDB**: 문서 청킹 + 임베딩 + ChromaDB 구축
- **NewsRAG**: 전체 RAG 파이프라인 + 대화 관리

---

## 사용 예시

### 1. 단일 주제 검색

```
검색 쿼리: AI 기술
```

### 2. 여러 주제 동시 검색

```
검색 쿼리: AI 기술, LLM, 생성형 AI
```

### 3. 프로그래밍 방식 사용

```python
from lib.naver_news_rag import NewsRAG
import os

# RAG 시스템 초기화
rag = NewsRAG(
    naver_client_id=os.getenv("NAVER_CLIENT_ID"),
    naver_client_secret=os.getenv("NAVER_CLIENT_SECRET"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    use_openai_embedding=False,
    llm_model="gpt-4o-mini",
    temperature=0.1
)

# 단일 주제 검색 (최신순, 기본값)
rag.build_vector_db("AI 기술", num_links=10, sort="date")

# 여러 주제 동시 검색 (최신순)
rag.build_vector_db(["AI 기술", "LLM", "생성형 AI"], num_links=10, sort="date")

# 유사도순 검색
rag.build_vector_db("AI 기술", num_links=10, sort="sim")

# 대화
answer = rag.chat("AI 최신 동향은?")
print(answer)

# 대화 히스토리 확인
rag.show_history()

# 대화 히스토리 초기화
rag.clear_history()

# 캐시된 주제 목록
cached = rag.list_cached_queries()
print(cached)
```

## WebUI 기능

### 주제 관리
- **새 주제 검색**: 여러 주제를 쉼표로 구분하여 입력
- **정렬 방식 선택**: 최신순 또는 유사도순 선택 (기본값: 최신순)
- **뉴스 개수 설정**: 5개 ~ 30개 선택 가능
- **캐시 새로고침**: 캐시된 주제 목록 갱신
- **캐시 목록**: 현재 세션에 캐시된 주제 확인

### 대화 관리
- **히스토리 보기**: 이전 대화 내용 확인
- **히스토리 초기화**: 대화 기록 삭제
- **전체 캐시 삭제**: 모든 캐시된 주제 삭제

### 상태 표시
- **현재 주제**: 현재 활성화된 검색 주제
- **대화 횟수**: 누적 대화 개수
- **캐시 개수**: 캐시된 주제 개수

## System Architecture

### 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 (User)                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    웹 인터페이스 (WebUI)                          │
│                    templates/index.html                          │
│                  - 다크모드 UI                                    │
│                  - 실시간 채팅 인터페이스                          │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP/REST API
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Flask 웹 서버 (webui.py)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ REST API 엔드포인트                                        │   │
│  │ - POST /api/build_db    : Vector DB 구축                 │   │
│  │ - POST /api/chat        : 챗봇 대화                       │   │
│  │ - GET  /api/history     : 대화 히스토리 조회               │   │
│  │ - POST /api/history/clear : 히스토리 초기화               │   │
│  │ - GET  /api/cache       : 캐시 목록 조회                  │   │
│  │ - POST /api/cache/clear : 캐시 삭제                       │   │
│  │ - GET  /api/status      : 시스템 상태 조회                 │   │
│  │ - POST /api/change_model : LLM 모델 변경                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  세션 관리: 세션별 RAG 인스턴스 격리 (30일 유지)                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              RAG 시스템 (lib/naver_news_rag.py)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ NewsRAG (통합 관리자)                                      │   │
│  │ - build_vector_db()  : Vector DB 구축 및 캐시 관리         │   │
│  │ - chat()            : 대화형 챗봇 응답                      │   │
│  │ - 대화 히스토리 관리 (최대 10개)                            │   │
│  │ - 세션 메모리 캐시 (db_cache)                               │   │
│  └────┬────────────────────────────────┬────────────────────┘   │
│       │                                │                        │
│       ▼                                ▼                        │
│  ┌─────────────────┐         ┌─────────────────────────┐        │
│  │ NaverNewsParser │         │ NewsVectorDB            │        │
│  │                 │         │                         │        │
│  │ - API 검색      │         │ - 문서 청킹             │        │
│  │ - HTML 파싱     │         │ - 임베딩 생성           │        │
│  │ - 텍스트 정제   │         │ - ChromaDB 구축         │        │
│  └────┬────────────┘         └────┬────────────────────┘        │
└───────┼──────────────────────────┼─────────────────────────────┘
        │                          │
        ▼                          ▼
┌────────────────────┐    ┌──────────────────────────┐
│  외부 API 서비스    │    │  LangChain Components    │
│                    │    │                          │
│ - 네이버 뉴스 API   │    │ - HuggingFace Embeddings│
│   (검색)           │    │   (jhgan/ko-sroberta-nli)│
│                    │    │ - OpenAI Embeddings     │
│ - OpenAI API       │    │   (text-embedding-3)    │
│   (LLM/임베딩)      │    │ - ChatOpenAI            │
└────────────────────┘    │   (gpt-4o-mini, etc.)   │
                          │ - ChromaDB (벡터 저장소) │
                          │ - RecursiveTextSplitter │
                          └──────────────────────────┘
```

### 데이터 흐름

1. **뉴스 검색 및 파싱**
   ```
   사용자 쿼리 → NaverNewsParser.get_news_links()
   → 네이버 API 호출 (최신순/유사도순)
   → HTML 다운로드 → BeautifulSoup 파싱
   → 제목 + 본문 추출 → Document 객체 생성
   ```

2. **Vector DB 구축**
   ```
   Document 리스트 → RecursiveTextSplitter (청킹)
   → HuggingFace/OpenAI Embeddings (벡터 변환)
   → ChromaDB (벡터 저장) → 세션 메모리 캐시
   ```

3. **RAG 질의 응답**
   ```
   사용자 질문 → ChromaDB Retriever (유사도 검색)
   → 관련 문서 검색 → 대화 히스토리 결합
   → ChatPromptTemplate → OpenAI LLM
   → 답변 생성 → 히스토리 저장 → 사용자 반환
   ```

### 주요 컴포넌트

#### 1. **NaverNewsParser** (뉴스 파서)
- **역할**: 네이버 뉴스 검색 및 본문 파싱
- **입력**: 검색 쿼리, 뉴스 개수, 정렬 방식
- **출력**: LangChain Document 객체 리스트
- **기능**:
  - 네이버 뉴스 API 호출 (최신순/유사도순)
  - HTML 파싱 (BeautifulSoup)
  - 제목 + 본문만 추출
  - 광고/노이즈 텍스트 자동 제거

#### 2. **NewsVectorDB** (벡터 데이터베이스)
- **역할**: 문서를 벡터로 변환하여 검색 가능한 DB 구축
- **입력**: Document 리스트
- **출력**: ChromaDB Retriever
- **기능**:
  - RecursiveCharacterTextSplitter로 청킹 (2000자, 200자 오버랩)
  - HuggingFace 임베딩 (jhgan/ko-sroberta-nli) 또는 OpenAI 임베딩
  - ChromaDB에 벡터 저장 (메모리 기반)
  - Retriever 제공 (유사도 검색)

#### 3. **NewsRAG** (RAG 시스템 통합)
- **역할**: 전체 RAG 파이프라인 관리 및 대화형 챗봇
- **입력**: 검색 쿼리, 사용자 질문
- **출력**: AI 답변
- **기능**:
  - Vector DB 구축 및 세션 캐시 관리
  - 대화 히스토리 관리 (최대 10개)
  - LangChain 기반 RAG 체인 구성
  - OpenAI LLM으로 답변 생성
  - 맥락 기반 대화 지원

#### 4. **Flask WebUI** (웹 서버)
- **역할**: REST API 서버 및 웹 인터페이스 제공
- **입력**: HTTP 요청 (JSON)
- **출력**: JSON 응답
- **기능**:
  - 세션별 RAG 인스턴스 격리
  - 세션 유지 (30일)
  - REST API 엔드포인트 제공
  - 에러 핸들링 및 검증

### 캐시 및 세션 관리

- **세션 격리**: Flask 세션 ID별로 독립적인 RAG 인스턴스 생성
- **Vector DB 캐시**:
  - 검색 쿼리 → 정렬된 주제명으로 캐시 키 생성
  - 메모리 기반 (프로그램 종료 시 자동 삭제)
  - 동일 주제 재검색 시 캐시 재사용
- **대화 히스토리**: 세션별 최대 10개 대화 저장

### 환경 구성

- **check_env.py**: API 키 검증 및 .env 파일 진단
- **.env**: 환경 변수 저장 (API 키)
- **requirements.txt**: Python 패키지 의존성

## 프로젝트 구조

```
news_parse/
├── lib/
│   ├── __init__.py           # 모듈 초기화
│   └── naver_news_rag.py     # RAG 시스템 메인 모듈
├── templates/
│   └── index.html            # WebUI 템플릿 (다크모드)
├── webui.py                  # Flask 서버
├── check_env.py              # 환경 변수 검증 스크립트
├── requirements.txt          # 패키지 의존성
├── .env.example              # 환경 변수 예시
├── .env                      # 환경 변수 (직접 생성 필요)
├── .gitignore                # Git 제외 파일
└── README.md                 # 프로젝트 문서
```

## 주요 클래스

### NaverNewsParser
네이버 뉴스 API 검색 및 HTML 파싱

```python
parser = NaverNewsParser(client_id, client_secret)
links = parser.get_news_links("AI 기술", num_links=10)
docs = parser.get_news_documents(links)
```

### NewsVectorDB
벡터 데이터베이스 구축 및 관리

```python
vector_db = NewsVectorDB(use_openai=False)
db = vector_db.build_from_documents(docs)
retriever = vector_db.get_retriever()
```

### NewsRAG
전체 RAG 시스템 (파서 + Vector DB + LLM)

```python
rag = NewsRAG(
    naver_client_id=client_id,
    naver_client_secret=client_secret,
    openai_api_key=api_key
)
rag.build_vector_db("AI 기술", num_links=10)
answer = rag.chat("AI 최신 동향은?")
```

## LLM 모델 선택

다음 모델들을 지원합니다:

- `gpt-4o-mini` (기본, 빠르고 저렴)
- `gpt-4o` (높은 품질)
- `gpt-3.5-turbo` (가장 저렴)
- `gpt-4-turbo` (균형잡힌 성능)

## 임베딩 모델

- **HuggingFace** (기본): jhgan/ko-sroberta-nli - 무료, 한국어 특화
- **OpenAI**: text-embedding-3-small - 유료, 높은 품질

## 캐시 동작 방식

- Vector DB는 **세션 메모리**에 캐시됩니다
- 프로그램 종료 시 자동 삭제
- 동일한 주제 재검색 시 캐시된 DB 재사용 가능
- 여러 주제를 검색한 경우 정렬된 주제명으로 캐시 키 생성

## 뉴스 검색 및 파싱

### 검색 방식
- **정렬**: 최신순(date) 또는 유사도순(sim) 선택 가능
- **기본 설정**: 최신순 정렬로 항상 최신 뉴스 우선 제공
- **개수**: 주제당 5개 ~ 30개 뉴스 검색 가능

### 파싱 범위
- **제목**: 뉴스 기사 제목만 추출
- **본문**: 뉴스 기사 본문만 추출
- **메타데이터**: URL, 제목

불필요한 광고, 내비게이션, 댓글 등은 자동으로 제거됩니다.

## 대화 히스토리

- 최대 10개의 대화 내용 저장
- 이전 대화를 기반으로 맥락 유지
- 새로운 주제 검색 시 히스토리 초기화 옵션 제공

## 문제 해결

### ImportError 발생 시

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU 관련 에러

```bash
# CPU 버전 PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Meta Tensor 오류 (Cannot copy out of meta tensor)

HuggingFace 임베딩 모델 로드 시 발생하는 경우:

**해결 방법 1: 패키지 재설치**
```bash
pip uninstall torch transformers sentence-transformers -y
pip install torch>=2.0.0,<2.5.0
pip install transformers>=4.35.0,<4.50.0
pip install sentence-transformers>=2.2.0,<4.0.0
```

**해결 방법 2: OpenAI 임베딩 사용**
`.env` 파일에서 임베딩 모델을 OpenAI로 변경:
```python
# webui.py 또는 CLI에서 OpenAI 임베딩 선택
use_openai_embedding=True
```

**해결 방법 3: 캐시 삭제**
```bash
rm -rf ~/.cache/huggingface/
```

### 네이버 API 에러

**401 Unauthorized 오류 (가장 흔한 오류)**

```
오류: 네이버 뉴스 API 요청 실패: 401 Client Error: Unauthorized
```

이 오류는 **네이버 API 키가 잘못되었거나 올바르게 설정되지 않았을 때** 발생합니다.

**해결 방법:**

1. **네이버 개발자 센터에서 API 키 확인**
   ```bash
   # https://developers.naver.com/apps/#/myapps 접속
   # 내 애플리케이션 클릭
   # Client ID와 Client Secret 복사
   ```

2. **프로젝트 루트에 .env 파일 생성**
   ```bash
   # .env.example 파일을 복사
   cp .env.example .env

   # 또는 직접 생성
   nano .env
   ```

3. **.env 파일에 API 키 입력 (따옴표 없이!)**
   ```env
   NAVER_CLIENT_ID=abc123xyz
   NAVER_CLIENT_SECRET=ABC123XYZ
   OPENAI_API_KEY=sk-abc123xyz
   ```

   **주의사항:**
   - ❌ 잘못된 예: `NAVER_CLIENT_ID="abc123xyz"` (따옴표 사용)
   - ❌ 잘못된 예: `NAVER_CLIENT_ID='abc123xyz'` (작은따옴표 사용)
   - ❌ 잘못된 예: `NAVER_CLIENT_ID= abc123xyz` (앞에 공백)
   - ✅ 올바른 예: `NAVER_CLIENT_ID=abc123xyz`

4. **검색 API 등록 확인**
   - 네이버 개발자 센터 > 내 애플리케이션
   - "사용 API" 섹션에 **"검색"** API가 있는지 확인
   - 없다면 "API 설정" 버튼 클릭 후 "검색" 체크

5. **서버 재시작**
   ```bash
   # Ctrl+C로 서버 종료 후 다시 실행
   python webui.py
   ```

6. **API 키 확인 (디버깅)**
   - 서버 시작 시 콘솔에 다음과 같이 표시됩니다:
   ```
   [API 키 확인]
   NAVER_CLIENT_ID: abc123xyz0... (길이: 16)
   NAVER_CLIENT_SECRET: ABC12... (길이: 16)
   ```
   - 길이가 너무 짧거나 길면 API 키가 잘못되었을 가능성이 높습니다

**'items' 오류 발생 시:**

```
KeyError: 'items'
```

이 오류는 네이버 API 응답에 검색 결과가 없을 때 발생합니다.

**원인 및 해결 방법:**

1. **API 사용량 초과**
   - [네이버 개발자 센터](https://developers.naver.com/apps/#/myapps) 접속
   - 내 애플리케이션 > 통계 메뉴에서 일일 사용량 확인
   - 일일 한도: 25,000건/일

2. **네트워크 오류**
   - 인터넷 연결 상태 확인
   - 방화벽 설정 확인

### OpenAI API 에러

- API 키 유효성 확인
- 크레딧 잔액 확인
- 모델 사용 권한 확인
