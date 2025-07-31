import pandas as pd

# 파일 경로
INPUT_CSV = "patent_data_final.CSV"
MAPPING_CSV = "manual_mapping.csv"
OUTPUT_XLSX = "출원인별_정규화_전체_소분류_중분류_개수.xlsx"

# 정규화 매핑 불러오기
mapping_df = pd.read_csv(MAPPING_CSV, encoding="utf-8-sig")
manual_mapping = dict(zip(mapping_df['raw_name'], mapping_df['normalized_name']))

def normalize_applicant(name):
    if pd.isna(name):
        return "UNKNOWN"
    name_std = str(name).strip()
    return manual_mapping.get(name_std, name_std)

# 입력 CSV 불러오기
df = pd.read_csv(INPUT_CSV, encoding="euc-kr")

# 필수 열 존재 여부 확인
if 'applicant' not in df.columns:
    raise ValueError("CSV 파일에 'applicant' 열이 존재하지 않습니다.")

# 결측 제거 및 출원인 정규화
df = df[df['applicant'].notna()]
df['normalized_applicant'] = df['applicant'].apply(normalize_applicant)

# 소분류/중분류가 있는 경우에만 추출
if 'label' in df.columns:
    df['subclass'] = df['label'].str.extract(r"([A-Z]{3})")
    df['midclass'] = df['subclass'].str[:2]
else:
    df['subclass'] = None
    df['midclass'] = None

# 전체 기준 출원인 수 계산
total_counts = df['normalized_applicant'].value_counts()

# 📢 상위 30개 출력
print("📊 출원인별 특허 개수 (상위 30개):\n")
print(total_counts.head(30))

# DataFrame으로 변환해 엑셀 저장용 준비
total_counts_df = total_counts.reset_index()
total_counts_df.columns = ['normalized_applicant', 'count']

# ExcelWriter 생성
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:

    wrote_sheet = False  # 시트 생성 여부 확인용

    # 1. 전체 시트 작성
    total_counts_df.to_excel(writer, sheet_name="전체", index=False)
    wrote_sheet = True

    # 2. 소분류별 시트 생성
    for subclass in df['subclass'].dropna().unique():
        if not isinstance(subclass, str) or subclass.strip() == "":
            continue
        sub_df = df[df['subclass'] == subclass]
        counts = sub_df['normalized_applicant'].value_counts().reset_index()
        counts.columns = ['normalized_applicant', 'count']
        if not counts.empty:
            sheet_name = subclass[:31]  # 엑셀 시트 이름 제한
            counts.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_sheet = True

    # 3. 중분류별 시트 생성
    for midclass in df['midclass'].dropna().unique():
        if not isinstance(midclass, str) or midclass.strip() == "":
            continue
        mid_df = df[df['midclass'] == midclass]
        counts = mid_df['normalized_applicant'].value_counts().reset_index()
        counts.columns = ['normalized_applicant', 'count']
        if not counts.empty:
            sheet_name = midclass[:31]
            counts.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_sheet = True

    # 4. 안전장치: 모든 시트 비어있을 경우
    if not wrote_sheet:
        empty_df = pd.DataFrame({"Message": ["No data available"]})
        empty_df.to_excel(writer, sheet_name="EMPTY", index=False)

print(f"\n✅ '{OUTPUT_XLSX}' 파일로 전체/소분류/중분류 기준 출원인 개수가 저장되었습니다.")
