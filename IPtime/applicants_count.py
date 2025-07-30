import pandas as pd

# CSV 파일 경로
INPUT_CSV = "applicant_count_input.CSV"
OUTPUT_NAME = "출원인_정규화_개수.txt"


# manual_mapping.csv 불러오기
mapping_df = pd.read_csv("manual_mapping.csv", encoding="utf-8-sig")

# 딕셔너리로 변환
manual_mapping = dict(zip(mapping_df['raw_name'], mapping_df['normalized_name']))

def normalize_applicant(name):
    if pd.isna(name):
        return "UNKNOWN"
    name_std = str(name).strip()
    return manual_mapping.get(name_std, name_std)

# CSV 불러오기
# encoding='cp949'를 추가하여 인코딩 문제 해결 시도
df = pd.read_csv(INPUT_CSV, encoding='euc-kr')

# applicant 열이 없는 경우 예외 처리
if 'applicant' not in df.columns:
    raise ValueError("CSV 파일에 'applicant' 열이 존재하지 않습니다.")

# NaN은 제거 (count에서 제외)
df = df[df['applicant'].notna()]

# 출원인별 특허 수 세기
df['normalized_applicant'] = df['applicant'].apply(normalize_applicant)
applicant_counts = df['normalized_applicant'].value_counts()

# 결과 출력 (상위 30개만 보기)
print("📊 출원인별 특허 개수 (상위 30개):\n")
print(applicant_counts.head(30))


# 전체 결과 출력
#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', 1000)
#pd.set_option('display.width', 1000)
#pd.set_option('display.unicode.east_asian_width', True)

#print("📊 출원인별 특허 개수 (전체 출력):\n")
#print(applicant_counts.to_string())

# Series → DataFrame 변환
applicant_counts_df = applicant_counts.reset_index()
applicant_counts_df.columns = ['normalized_name', 'count']

# TXT 파일로 저장
with open(OUTPUT_NAME, "w", encoding="utf-8") as f:
    f.write("📊 정규화된 출원인별 특허 개수\n\n")
    for _, row in applicant_counts_df.iterrows():
        f.write(f"{row['normalized_name']} : {row['count']}\n")

print(f"\n✅ '{OUTPUT_NAME}'로 저장되었습니다.")