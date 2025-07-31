import pandas as pd
from collections import defaultdict

# 엑셀 불러오기
xls = pd.ExcelFile("fixed2 라벨별_시트분리.xlsx", engine="openpyxl") # 파일 경로
sheet_names = xls.sheet_names

# 핵심 점수만 포함
valid_scores = {"S", "A"}

# 연도 -> 시기 매핑 함수
def get_period(year):
    if pd.isna(year):
        return None
    year = int(year)
    if year <= 2015:
        return "~2015"
    elif 2016 <= year <= 2020:
        return "2016–2020"
    elif 2021 <= year <= 2022:
        return "2021–2022"
    elif 2023 <= year <= 2025:
        return "2023–2025"
    else:
        return "기타"

# 결과 테이블 저장용
flow_rows = []

for sheet in sheet_names:
    df = xls.parse(sheet)
    
    if "application_date" not in df.columns or "core_patent_score" not in df.columns:
        continue
    
    # 날짜 처리 및 연도 추출
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["year"] = df["application_date"].dt.year
    
    # 핵심 특허만 필터
    core_df = df[df["core_patent_score"].isin(valid_scores)].copy()
    core_df["period"] = core_df["year"].apply(get_period)

    # 시기별 그룹화 및 대표 특허 ID 추출
    for period, group in core_df.groupby("period"):
        ids = group.sort_values("core_patent_score").head(3)["id"].tolist()
        flow_rows.append({
            "소분류 코드": sheet,
            "시기": period,
            "대표 특허 ID": ", ".join(map(str, ids)),
            "핵심 특허 수": len(group)
        })

# DataFrame으로 변환
flow_df = pd.DataFrame(flow_rows)

# 엑셀 저장
flow_df.to_excel("기술흐름_분석_결과.xlsx", index=False)
print("기술 흐름 엑셀 저장 완료!")
