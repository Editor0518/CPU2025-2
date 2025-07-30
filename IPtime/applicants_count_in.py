import pandas as pd

# 파일 경로
INPUT_CSV = "applicant_count_input.CSV"
MAPPING_CSV = "manual_mapping.csv"
OUTPUT_XLSX = "출원인별_정규화_소분류_중분류_개수.xlsx"

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

# 필요한 열 확인
if 'applicant' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV 파일에 'applicant' 또는 'label' 열이 존재하지 않습니다.")

# 결측 제거 및 정규화
df = df[df['applicant'].notna()]
df['normalized_applicant'] = df['applicant'].apply(normalize_applicant)

# 소분류 및 중분류 추출
# 소분류 및 중분류 추출 (괄호 제거 후 3자리 코드만 추출)
df['subclass'] = df['label'].str.extract(r"([A-Z]{3})")
df['midclass'] = df['subclass'].str[:2]


# ExcelWriter 생성
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:

    wrote_sheet = False  # 시트 하나라도 쓰였는지 확인

    # 1. 소분류별 시트 생성
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

    # 2. 중분류별 시트 생성
    for midclass in df['midclass'].dropna().unique():
        if not isinstance(midclass, str) or midclass.strip() == "":
            continue
        mid_df = df[df['midclass'] == midclass]
        counts = mid_df['normalized_applicant'].value_counts().reset_index()
        counts.columns = ['normalized_applicant', 'count']
        if not counts.empty:
            sheet_name = midclass[:31]  # 시트 이름 제한
            counts.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_sheet = True

    # 마지막 안전장치: 시트 하나도 안 써졌으면 빈 시트라도 하나 생성
    if not wrote_sheet:
        empty_df = pd.DataFrame({"Message": ["No data available"]})
        empty_df.to_excel(writer, sheet_name="EMPTY", index=False)

print(f"✅ '{OUTPUT_XLSX}' 파일로 소분류/중분류별 출원인 개수가 저장되었습니다.")
