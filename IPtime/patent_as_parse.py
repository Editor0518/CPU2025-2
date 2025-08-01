import pandas as pd

# 파일 경로
INPUT_CSV = "patent_data_final.CSV"
OUTPUT_XLSX = "fixed2 라벨별_시트분리.xlsx"

# CSV 불러오기 (인코딩은 실제 파일에 따라 맞춰 주세요)
df = pd.read_csv(INPUT_CSV, encoding="cp949")

# label 컬럼이 존재하는지 확인
if 'label' not in df.columns:
    raise ValueError("CSV 파일에 'label' 컬럼이 존재하지 않습니다.")

# ExcelWriter 생성
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    label_list = df['label'].dropna().unique()
    if len(label_list) == 0:
        # 아무 데이터도 없으면 빈 시트 작성
        pd.DataFrame({"Message": ["No label data found"]}).to_excel(writer, sheet_name="EMPTY", index=False)
    else:
        for label in label_list:
            # 각 label별 데이터 필터링
            label_df = df[df['label'] == label]
            if not label_df.empty:
                sheet_name = str(label)[:31]  # 엑셀 시트 이름 제한
                label_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ '{OUTPUT_XLSX}' 파일에 라벨별로 시트가 분리되어 저장되었습니다.")
