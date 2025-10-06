# -*- coding: utf-8 -*-
"""
네이버 뉴스 RAG 챗봇 라이브러리
Python 3.11+ 호환

특징:
- 네이버 뉴스를 기반으로 정보를 제공하는 대화형 AI 챗봇
- Vector DB는 세션 메모리에 캐시됩니다 (프로그램 종료 시 삭제)
- 대화 히스토리를 유지하며 컨텍스트 기반 대화 가능
- 여러 주제를 메모리에 캐싱하여 빠른 전환 가능

.env 파일 설정 방법:
1. 프로젝트 루트에 .env 파일 생성
2. 아래 내용 입력:
   NAVER_CLIENT_ID=your_naver_client_id
   NAVER_CLIENT_SECRET=your_naver_client_secret
   OPENAI_API_KEY=your_openai_api_key

사용 예시:
    from lib import NewsRAG, main

    # 대화형 인터페이스 실행
    main()

    # 프로그래밍 방식 사용
    rag = NewsRAG(client_id, client_secret)
    rag.build_vector_db("AI 기술", num_links=10)
    answer = rag.chat("AI 최신 동향은?")
"""

from lib.naver_news_rag import (
    NaverNewsParser,
    NewsVectorDB,
    NewsRAG,
    main
)

__version__ = "1.0.0"

__all__ = [
    "NaverNewsParser",
    "NewsVectorDB",
    "NewsRAG",
    "main",
]
