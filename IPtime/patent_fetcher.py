import requests
import xmltodict
import pandas as pd
import math
from pathlib import Path

def fetch_patents(
    search_word: str,
    year_to_search: str = '0',
    api_key_path: str = './apiKey.txt', #KIPRIS API 키 파일 경로
    output_tsv: str = './patent-data/patents_for_bert.tsv',
    rows_per_page: int = 10,
    withdrawn: bool = False
) -> pd.DataFrame:
    url = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch"
    api_key = Path(api_key_path).read_text(encoding='utf-8').strip()
    all_items = []

    # 상태 코드와 라벨
    STATUS_LABELS = {
        "A": "공개 특허(A)",
        "R": "등록 특허(R)",
        "C": "취하 특허(C)",
        "F": "소멸 특허(F)",
        "G": "포기 특허(G)",
        "": "전체"
    }

    # 요청 상태 코드
    if withdrawn:
        lastvalues = [""]
    else:
        lastvalues = ["A", "R"]

    for lv in lastvalues:
        label = STATUS_LABELS.get(lv, lv)
        print(f"🔎 {label} 데이터 수집 시작...")

        def _get_params(page_no):
            return {
                "word": search_word,
                "year": year_to_search,
                "patent": True,
                "utility": False,
                "ServiceKey": api_key,
                "numOfRows": rows_per_page,
                "pageNo": page_no,
                "lastvalue": lv
            }

        resp = requests.get(url, params=_get_params(1))
        #print(f"▶ API URL: {resp.url}")  # 디버그용 URL 출력
        resp.raise_for_status()
        doc = xmltodict.parse(resp.text)
        total_count = int(doc['response']['count']['totalCount'])
        total_pages = math.ceil(total_count / rows_per_page)
        print(f"✔ {label}: {total_count}건 (총 {total_pages} 페이지)")

        for page_no in range(1, total_pages + 1):
            resp = requests.get(url, params=_get_params(page_no))
            resp.raise_for_status()
            items = xmltodict.parse(resp.text)['response']['body']['items'].get('item', [])
            if isinstance(items, dict):
                items = [items]
            all_items.extend(items)

    # DataFrame 생성 + 중복 제거
    df = pd.DataFrame(all_items)
    if 'applicationNumber' in df.columns:
        df = df.drop_duplicates(subset=['applicationNumber'])

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"✔ 최종 {len(df)}건 수집 완료 — 저장된 파일: {output_tsv}")
    return df
