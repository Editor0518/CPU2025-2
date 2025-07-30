# citation_fetcher.py

import requests
from xml.etree import ElementTree as ET
from pathlib import Path

# API 키 파일 경로
API_KEY_PATH = './apiKey.txt'

# API 키 읽기
def read_service_key():
    try:
        return Path(API_KEY_PATH).read_text(encoding='utf-8').strip()
    except Exception as e:
        raise RuntimeError(f"[오류] API 키를 읽을 수 없습니다: {e}")

# 피인용 횟수 조회 함수
def fetch_citation_count(application_number: str) -> int:
    """
    주어진 출원번호(application_number)에 대해 피인용 횟수를 반환합니다.
    """
    service_key = read_service_key()
    url = "http://plus.kipris.or.kr/openapi/rest/CitingService/citingInfo"
    params = {
        'standardCitationApplicationNumber': application_number,
        'serviceKey': service_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"[HTTP 오류] {application_number} → 상태코드: {response.status_code}")
            return 0

        root = ET.fromstring(response.text)
        citing_list = root.findall(".//citingInfo")
        return len(citing_list)

    except Exception as e:
        print(f"[예외 발생] {application_number} 처리 중 오류: {e}")
        return 0
