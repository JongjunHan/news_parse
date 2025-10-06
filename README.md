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

## 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트 및 기능 개선 제안을 환영합니다.

## 연락처

문의사항이 있으시면 이슈를 등록해주세요.
