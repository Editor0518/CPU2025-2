import pandas as pd

# 파일 경로
MAPPING_CSV = "manual_mapping.csv"

INPUT_CSV = "patent_data_final.CSV"
OUTPUT_XLSX = "출원기업별세부분석.xlsx"

# 대상 출원인
target_applicants = [
    "SAMSUNG ELECTRONICS",
    "TENCENT",
    "LG ELECTRONICS",
    "BAIDU",
    "ADOBE",
    "MICROSOFT",
    "GOOGLE",
    "NVIDIA"
]

# 연도 범위 설정
years = list(range(2005, 2026))  # 2005 ~ 2025

# 데이터 로드
df = pd.read_csv(INPUT_CSV, encoding="cp949")

# 날짜 파싱
df['application_date'] = pd.to_datetime(df['application_date'], format='%Y.%m.%d', errors='coerce')
df['year'] = df['application_date'].dt.year

# 소분류, 중분류 추출
df['소분류'] = df['label'].astype(str)
df['중분류'] = df['소분류'].str[:2]

# 출원인 정규화
mapping_df = pd.read_csv(MAPPING_CSV, encoding="utf-8-sig")
manual_mapping = dict(zip(mapping_df['raw_name'], mapping_df['normalized_name']))
df['applicant'] = df['applicant'].map(manual_mapping).fillna(df['applicant'])

# 출원인 필터링
df = df[df['applicant'].isin(target_applicants)]

# 연도 필터링
df = df[df['year'].between(2005, 2025)]

# 엑셀 저장
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    for applicant in target_applicants:
        app_df = df[df['applicant'] == applicant]

        # ✅ 1. 소분류 기준 피벗 테이블
        pivot_small = pd.pivot_table(
            app_df,
            index='소분류',
            columns='year',
            values='applicant',
            aggfunc='count',
            fill_value=0
        )

        # 누락된 연도 채우기
        for year in years:
            if year not in pivot_small.columns:
                pivot_small[year] = 0
        pivot_small = pivot_small[sorted(pivot_small.columns)]  # 연도 순 정렬

        # 시트 저장
        sheet_name_small = f"{applicant[:25]}_소분류"
        pivot_small.to_excel(writer, sheet_name=sheet_name_small)

        # ✅ 2. 중분류 기준 피벗 테이블
        pivot_mid = pd.pivot_table(
            app_df,
            index='중분류',
            columns='year',
            values='applicant',
            aggfunc='count',
            fill_value=0
        )

        for year in years:
            if year not in pivot_mid.columns:
                pivot_mid[year] = 0
        pivot_mid = pivot_mid[sorted(pivot_mid.columns)]

        sheet_name_mid = f"{applicant[:25]}_중분류"
        pivot_mid.to_excel(writer, sheet_name=sheet_name_mid)

print(f"✅ 완료: '{OUTPUT_XLSX}' 파일에 출원인별로 [소분류, 중분류] 시트가 각각 저장되었습니다.")
