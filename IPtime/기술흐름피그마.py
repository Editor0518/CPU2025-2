import pandas as pd
import re

def get_period(year):
    if pd.isna(year):
        return None
    bins = list(range(2000, 2030, 5))  # 2000~2025까지 5년 단위
    for i in range(len(bins) - 1):
        if bins[i] <= year < bins[i + 1]:
            return f"{bins[i]}~{bins[i + 1] - 1}"
    return "기타"

# 엑셀 파일 경로
file_path = "fixed2 라벨별_시트분리.xlsx"
xls = pd.ExcelFile(file_path, engine="openpyxl")

rows = []

for sheet in xls.sheet_names:
    df = xls.parse(sheet)
    
    if "application_date" not in df.columns or "core_patent_score" not in df.columns:
        continue

    df["application_date"] = pd.to_datetime(df["application_date"], errors='coerce')
    df["year"] = df["application_date"].dt.year
    df["period"] = df["year"].apply(get_period)

    filtered = df[df["core_patent_score"].isin(["S", "A"])].copy()
    filtered = filtered.sort_values(by=["core_patent_score", "year"], ascending=[True, True])

    for _, row in filtered.iterrows():
        rows.append({
            "소분류": sheet,
            "기간": row.get("period", ""),
            "출원일": row.get("application_date", ""),
            "특허 ID": row.get("patent_id", ""),
            "요약": row.get("summary", "") or row.get("invention_title", ""),
            "점수": row.get("core_patent_score", ""),
        })

# 결과 저장
flow_df = pd.DataFrame(rows)
flow_df.to_csv("기술흐름도_Figma_입력용.csv", index=False, encoding="utf-8-sig")
print("CSV 저장 완료!")
