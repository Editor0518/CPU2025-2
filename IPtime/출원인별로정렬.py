import pandas as pd

# 파일 경로
INPUT_CSV = "생성형AI 기술흐름도 데이터 (1).CSV"
OUTPUT_XLSX = "출원날짜_라벨별_시트분리.xlsx"

# CSV 불러오기
df = pd.read_csv(INPUT_CSV, encoding="cp949")

# label 컬럼이 존재하는지 확인
if 'label' not in df.columns:
    raise ValueError("CSV 파일에 'label' 컬럼이 존재하지 않습니다.")

# 출원일이 날짜형식이 아니면 변환
if not pd.api.types.is_datetime64_any_dtype(df['출원일']):
    df['출원일'] = pd.to_datetime(df['출원일'], errors='coerce')

# 중분류 컬럼 생성 (label 앞 2자리)
df['중분류'] = df['label'].astype(str).str[:2]

with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    # ① 소분류(label)별 저장
    label_list = df['label'].dropna().unique()
    if len(label_list) == 0:
        pd.DataFrame({"Message": ["No label data found"]}).to_excel(writer, sheet_name="EMPTY_LABEL", index=False)
    else:
        for label in label_list:
            label_df = df[df['label'] == label].sort_values(by='출원일', ascending=False)
            if not label_df.empty:
                sheet_name = f"L_{str(label)[:28]}"  # 'L_' 붙이고 시트이름 최대 31자 제한
                label_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # ② 중분류별 저장
    mid_label_list = df['중분류'].dropna().unique()
    for mid_label in mid_label_list:
        mid_df = df[df['중분류'] == mid_label].sort_values(by='출원일', ascending=False)
        if not mid_df.empty:
            sheet_name = f"M_{str(mid_label)[:29]}"  # 'M_' 붙이고 시트이름 제한
            mid_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ '{OUTPUT_XLSX}' 파일에 라벨별(L_) 및 중분류별(M_)로 시트가 분리되어 저장되었습니다.")
