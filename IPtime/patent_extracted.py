import pandas as pd
import os
from patent_fetcher import fetch_patents
from citation_fetcher import fetch_citation_count

print("KIPRIS 특허 데이터 추출 프로그램")

# 사용자 입력
search_word = input("검색식을 입력하세요: ").strip()
withdrawn_input = input("취하/소멸/포기 포함 여부 (y/n): ").strip().lower()
withdrawn_flag = withdrawn_input == 'y'

print(f"검색 중... (검색어: {search_word})")
df = fetch_patents(search_word=search_word, year_to_search="0", withdrawn=withdrawn_flag)

if df.empty:
    print("검색 결과가 없습니다.")
    exit(0)

print(f"[OK] {len(df)}건의 특허 데이터 수집 완료")

# 출원년도 추출 함수 (YYYY-MM-DD 형식 가정)
def extract_year(date_str):
    if pd.isna(date_str) or len(str(date_str)) < 4:
        return ""
    return str(date_str)[:4]

# 실제 존재하는 컬럼 기준으로 추출
df_extracted = pd.DataFrame()
df_extracted['발명명'] = df.get('inventionTitle', '')
df_extracted['요약'] = df.get('astrtCont', '')
df_extracted['IPC/CPC 코드'] = df.get('ipcNumber', '')
df_extracted['출원년도'] = df.get('applicationDate', '').apply(extract_year)
df_extracted['국가'] = 'KR'
df_extracted['청구항 1번'] = ''

# 피인용 횟수 계산
print("피인용 횟수 계산 중...")
df['applicationNumber'] = df['applicationNumber'].fillna('')
df_extracted['피인용횟수'] = df['applicationNumber'].apply(
    lambda num: fetch_citation_count(num) if num else 0
)

# 저장
output_file = "patents_extracted.csv"
df_extracted.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"[OK] 필요한 정보 + 피인용 횟수 추출 완료")
print(f"[OK] CSV 파일로 저장 완료: {output_file}")
print("프로그램 종료.")
