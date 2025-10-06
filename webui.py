# -*- coding: utf-8 -*-
"""
네이버 뉴스 RAG 챗봇 WebUI
Flask 기반 웹 인터페이스
"""

import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from lib.naver_news_rag import NewsRAG
import secrets

# .env 파일 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # 세션 암호화 키

# 세션 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 30  # 30일 유지
app.config['SESSION_PERMANENT'] = True

# 세션별 RAG 인스턴스 저장소
rag_instances = {}

def get_rag_instance():
    """현재 세션의 RAG 인스턴스 가져오기 또는 생성"""
    session_id = session.get('session_id')

    if not session_id:
        session_id = secrets.token_hex(16)
        session['session_id'] = session_id
        session.permanent = True

    if session_id not in rag_instances:
        # 새로운 RAG 인스턴스 생성
        naver_client_id = os.getenv("NAVER_CLIENT_ID")
        naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # API 키 검증 및 디버깅 정보
        print("\n[API 키 확인]")
        if not naver_client_id:
            raise ValueError("NAVER_CLIENT_ID가 .env 파일에 설정되지 않았습니다.")
        if not naver_client_secret:
            raise ValueError("NAVER_CLIENT_SECRET가 .env 파일에 설정되지 않았습니다.")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # API 키 앞부분만 출력 (보안)
        print(f"NAVER_CLIENT_ID: {naver_client_id[:10]}... (길이: {len(naver_client_id)})")
        print(f"NAVER_CLIENT_SECRET: {naver_client_secret[:5]}... (길이: {len(naver_client_secret)})")
        print(f"OPENAI_API_KEY: {openai_api_key[:10]}... (길이: {len(openai_api_key)})")

        # 공백 확인
        naver_client_id = naver_client_id.strip()
        naver_client_secret = naver_client_secret.strip()
        openai_api_key = openai_api_key.strip()

        rag_instances[session_id] = NewsRAG(
            naver_client_id=naver_client_id,
            naver_client_secret=naver_client_secret,
            openai_api_key=openai_api_key,
            use_openai_embedding=False,
            llm_model="gpt-4o-mini",
            temperature=0.1
        )

    return rag_instances[session_id]


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/api/build_db', methods=['POST'])
def build_db():
    """Vector DB 구축"""
    try:
        data = request.json
        query = data.get('query')  # 문자열 또는 리스트
        queries = data.get('queries')  # 리스트 (선택)
        num_links = data.get('num_links', 10)
        use_cache = data.get('use_cache', True)
        sort = data.get('sort', 'date')  # 정렬 방식 (기본값: 최신순)

        # queries가 있으면 우선 사용, 없으면 query 사용
        if queries:
            query = queries

        if not query:
            return jsonify({'error': '검색 쿼리가 필요합니다.'}), 400

        # 쿼리가 리스트면 빈 배열 체크
        if isinstance(query, list) and len(query) == 0:
            return jsonify({'error': '검색 쿼리가 필요합니다.'}), 400

        rag = get_rag_instance()

        # 캐시 키 생성 (lib/naver_news_rag.py와 동일한 방식)
        if isinstance(query, str):
            cache_key = query
        else:
            cache_key = " | ".join(sorted(query))

        # 캐시 확인
        if use_cache and cache_key in rag.list_cached_queries():
            rag.build_vector_db(query, num_links, use_cache=True, sort=sort)
            return jsonify({
                'success': True,
                'message': f'캐시된 Vector DB를 사용합니다: {cache_key}',
                'cached': True,
                'query': cache_key
            })
        else:
            rag.build_vector_db(query, num_links, use_cache=False, sort=sort)
            return jsonify({
                'success': True,
                'message': f'Vector DB 구축 완료: {cache_key}',
                'cached': False,
                'query': cache_key
            })

    except ValueError as e:
        # 네이버 API 오류 등 비즈니스 로직 에러
        return jsonify({'error': str(e), 'type': 'validation'}), 400
    except Exception as e:
        # 예상치 못한 서버 에러
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'서버 오류: {str(e)}', 'type': 'server'}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """챗봇 대화"""
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': '질문이 필요합니다.'}), 400

        rag = get_rag_instance()

        # Vector DB가 구축되어 있는지 확인
        if not rag.vector_db.db:
            return jsonify({'error': 'Vector DB가 구축되지 않았습니다. 먼저 주제를 검색해주세요.'}), 400

        answer = rag.chat(question)

        return jsonify({
            'success': True,
            'answer': answer,
            'history_count': len(rag.chat_history)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """대화 히스토리 가져오기"""
    try:
        rag = get_rag_instance()
        return jsonify({
            'success': True,
            'history': rag.chat_history,
            'count': len(rag.chat_history)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """대화 히스토리 초기화"""
    try:
        rag = get_rag_instance()
        rag.clear_history()

        return jsonify({
            'success': True,
            'message': '대화 히스토리가 초기화되었습니다.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache', methods=['GET'])
def get_cache():
    """캐시된 주제 목록 가져오기"""
    try:
        rag = get_rag_instance()
        cached_queries = rag.list_cached_queries()

        return jsonify({
            'success': True,
            'queries': cached_queries,
            'current': rag.current_query,
            'count': len(cached_queries)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """캐시 삭제"""
    try:
        data = request.json
        query = data.get('query')  # None이면 전체 삭제

        rag = get_rag_instance()
        rag.clear_cache(query)

        return jsonify({
            'success': True,
            'message': f"'{query}' 캐시가 삭제되었습니다." if query else '모든 캐시가 삭제되었습니다.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """현재 상태 가져오기"""
    try:
        rag = get_rag_instance()

        return jsonify({
            'success': True,
            'current_query': rag.current_query,
            'cached_queries': rag.list_cached_queries(),
            'history_count': len(rag.chat_history),
            'has_db': rag.vector_db.db is not None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/change_model', methods=['POST'])
def change_model():
    """LLM 모델 변경"""
    try:
        data = request.json
        model = data.get('model')

        if not model:
            return jsonify({'error': '모델 이름이 필요합니다.'}), 400

        # 허용된 모델 목록
        allowed_models = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo']
        if model not in allowed_models:
            return jsonify({'error': f'지원하지 않는 모델입니다. 허용된 모델: {", ".join(allowed_models)}'}), 400

        rag = get_rag_instance()
        rag.change_llm_model(model)

        return jsonify({
            'success': True,
            'message': f'LLM 모델이 {model}(으)로 변경되었습니다.',
            'model': model
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        """브라우저 자동 열기"""
        import time
        time.sleep(1.5)  # 서버 시작 대기
        webbrowser.open('http://localhost:5000')

    print("=" * 80)
    print("네이버 뉴스 RAG 챗봇 WebUI 시작")
    print("=" * 80)
    print("\n서버 정보:")
    print("  - URL: http://localhost:5000")
    print("  - 세션 유지: 30일")
    print("  - Ctrl+C로 종료")
    print("\n브라우저가 자동으로 열립니다...")
    print("=" * 80 + "\n")

    # 브라우저 자동 열기 (백그라운드 스레드)
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
