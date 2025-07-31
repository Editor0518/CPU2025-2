import pandas as pd

# 파일 경로
INPUT_CSV = "patent_data_final.CSV"
MAPPING_CSV = "manual_mapping.csv"
OUTPUT_XLSX = "출원인별_분류_카운트.xlsx"

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

# 'applicant' 열 필수 확인 및 정규화
if 'applicant' not in df.columns:
    raise ValueError("CSV 파일에 'applicant' 열이 존재하지 않습니다.")
df = df[df['applicant'].notna()]
df['normalized_applicant'] = df['applicant'].apply(normalize_applicant)

# 라벨에서 소분류, 중분류 추출
if 'label' in df.columns:
    df['subclass'] = df['label'].str.extract(r"([A-Z]{3})")  # AAA 등
    df['midclass'] = df['subclass'].str[:2]  # 앞 두 자리 추출 AA 등
else:
    df['subclass'] = None
    df['midclass'] = None

# 전체, 중분류, 소분류 기준 카운트
df['분류'] = '전체'
df_all = df[['normalized_applicant', '분류']].copy()

# 중분류 레코드
df_mid = df[['normalized_applicant', 'midclass']].dropna().copy()
df_mid.columns = ['normalized_applicant', '분류']

# 소분류 레코드
df_sub = df[['normalized_applicant', 'subclass']].dropna().copy()
df_sub.columns = ['normalized_applicant', '분류']

# 모두 합치기
df_cat = pd.concat([df_all, df_mid, df_sub], ignore_index=True)

# 피벗 테이블로 출원인별 분류별 개수 구하기
pivot_df = pd.pivot_table(
    df_cat,
    index='normalized_applicant',
    columns='분류',
    aggfunc='size',
    fill_value=0
).reset_index()

# 열 순서 정렬: '전체', 중분류(2글자), 소분류(3글자)
cols = pivot_df.columns.tolist()
cols_ordered = ['normalized_applicant']
if '전체' in cols:
    cols_ordered.append('전체')
mid_classes = sorted([c for c in cols if isinstance(c, str) and len(c) == 2 and c != '전체'])
sub_classes = sorted([c for c in cols if isinstance(c, str) and len(c) == 3])
cols_ordered += mid_classes + sub_classes
pivot_df = pivot_df[cols_ordered]

# '전체' 기준으로 내림차순 정렬
if '전체' in pivot_df.columns:
    pivot_df = pivot_df.sort_values(by='전체', ascending=False)

# 엑셀로 저장 (UTF-8 문자 깨짐 방지용 openpyxl 엔진 사용)
pivot_df.to_excel(OUTPUT_XLSX, index=False, engine='openpyxl')

print(f"\n✅ '{OUTPUT_XLSX}' 파일로 출원인별 전체/중분류/소분류 카운트가 저장되었습니다.")
