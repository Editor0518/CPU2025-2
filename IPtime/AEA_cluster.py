import pandas as pd

# 파일 경로
INPUT_FILE = "AEA_input.CSV"
OUTPUT_FILE = "AEA_matched_patents.xlsx"

# CSV 파일 불러오기
df = pd.read_csv(INPUT_FILE, encoding='cp949')

# 병합 텍스트 생성
df['combined_text'] = df[['title', 'korean_summary', 'summary', 'main_claim']].fillna('').agg(' '.join, axis=1)

# 검색 조건 판별 함수
def contains_keywords(text):
    text = text.lower()
    has_multi1 = '멀티모달' in text
    has_multi2 = '멀티미디어' in text
    has_all_three = all(x in text for x in ['오디오', '텍스트', '비디오'])
    return has_multi1 or has_multi2 or has_all_three

# 키워드 포함 여부 판단
df['has_keywords'] = df['combined_text'].apply(contains_keywords)

# 포함된 특허만 필터링
matched_df = df[df['has_keywords']]
unmatched_df = df[~df['has_keywords']]

# 포함된 수와 안 포함된 수 출력
print(f"✅ 키워드 포함 특허 수: {len(matched_df)}")
print(f"❌ 키워드 미포함 특허 수: {len(unmatched_df)}")

# 포함된 특허 ID 출력
print("🆔 포함된 특허 ID 목록:")
print(matched_df['id'].tolist())

# 엑셀로 저장
matched_df.to_excel(OUTPUT_FILE, index=False)

print(f"📄 포함된 특허 {len(matched_df)}건을 {OUTPUT_FILE} 로 저장했습니다.")
