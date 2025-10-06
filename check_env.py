#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.env 파일 검증 스크립트
API 키가 올바르게 설정되었는지 확인합니다.
"""

import os
from dotenv import load_dotenv

def check_env():
    """환경 변수 검증"""
    print("=" * 80)
    print(".env 파일 검증")
    print("=" * 80)

    # .env 파일 존재 확인
    if not os.path.exists('.env'):
        print("\n❌ .env 파일이 존재하지 않습니다!")
        print("\n해결 방법:")
        print("1. .env.example 파일을 복사하세요:")
        print("   cp .env.example .env")
        print("\n2. .env 파일을 열어서 API 키를 입력하세요:")
        print("   nano .env")
        return False

    print("\n✓ .env 파일 존재 확인")

    # .env 파일 로드
    load_dotenv()

    # API 키 확인
    all_ok = True

    print("\n[네이버 API 키 확인]")
    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

    if not naver_client_id or naver_client_id == "your_naver_client_id_here":
        print("❌ NAVER_CLIENT_ID가 설정되지 않았습니다.")
        all_ok = False
    else:
        # 따옴표 검사
        if naver_client_id.startswith('"') or naver_client_id.startswith("'"):
            print(f"⚠️  NAVER_CLIENT_ID에 따옴표가 포함되어 있습니다: {naver_client_id[:20]}...")
            print("   .env 파일에서 따옴표를 제거하세요.")
            all_ok = False
        else:
            print(f"✓ NAVER_CLIENT_ID: {naver_client_id[:10]}... (길이: {len(naver_client_id)})")

    if not naver_client_secret or naver_client_secret == "your_naver_client_secret_here":
        print("❌ NAVER_CLIENT_SECRET가 설정되지 않았습니다.")
        all_ok = False
    else:
        if naver_client_secret.startswith('"') or naver_client_secret.startswith("'"):
            print(f"⚠️  NAVER_CLIENT_SECRET에 따옴표가 포함되어 있습니다: {naver_client_secret[:10]}...")
            print("   .env 파일에서 따옴표를 제거하세요.")
            all_ok = False
        else:
            print(f"✓ NAVER_CLIENT_SECRET: {naver_client_secret[:5]}... (길이: {len(naver_client_secret)})")

    print("\n[OpenAI API 키 확인]")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        all_ok = False
    else:
        if openai_api_key.startswith('"') or openai_api_key.startswith("'"):
            print(f"⚠️  OPENAI_API_KEY에 따옴표가 포함되어 있습니다: {openai_api_key[:15]}...")
            print("   .env 파일에서 따옴표를 제거하세요.")
            all_ok = False
        else:
            print(f"✓ OPENAI_API_KEY: {openai_api_key[:15]}... (길이: {len(openai_api_key)})")

    # 최종 결과
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ 모든 API 키가 올바르게 설정되었습니다!")
        print("\n프로그램을 실행하세요:")
        print("  python webui.py")
    else:
        print("❌ API 키 설정에 문제가 있습니다.")
        print("\n해결 방법:")
        print("1. https://developers.naver.com/apps/#/myapps 에서 네이버 API 키 발급")
        print("2. https://platform.openai.com/api-keys 에서 OpenAI API 키 발급")
        print("3. .env 파일에 API 키 입력 (따옴표 없이)")
        print("\n예시:")
        print("  NAVER_CLIENT_ID=abc123xyz")
        print("  NAVER_CLIENT_SECRET=ABC123XYZ")
        print("  OPENAI_API_KEY=sk-abc123xyz")

    print("=" * 80)
    return all_ok


if __name__ == "__main__":
    check_env()
