import pandas as pd

# 파일 경로
INPUT_FILE = "patent_data_noise_apply.xlsx"
OUTPUT_FILE = "updated_patent_data_noise_apply.xlsx"

# 엑셀 파일의 모든 시트를 읽기 (openpyxl 사용)
xlsx = pd.read_excel(INPUT_FILE, sheet_name=None, engine='openpyxl')

# 필요한 시트 가져오기
df_total = xlsx['전체']
df_fix = xlsx['수정']

# 'id'를 기준으로 인덱스 설정
df_total.set_index('id', inplace=True)
df_fix.set_index('id', inplace=True)

# 'label' 컬럼만 업데이트
df_total.update(df_fix[['label']])

# 인덱스 복원
df_total.reset_index(inplace=True)
df_fix.reset_index(inplace=True)

# 다국어 깨짐 방지를 위한 엑셀 저장
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl', if_sheet_exists='replace', mode='w') as writer:
    df_total.to_excel(writer, sheet_name="전체", index=False)
    df_fix.to_excel(writer, sheet_name="수정", index=False)
