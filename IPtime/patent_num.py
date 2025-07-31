import pandas as pd
import numpy as np
import re
import chardet

INPUT_FILE_NAME = "patent_data_final.csv"
OUTPUT_FILE_NAME = "output_with_metrics.csv"

# manual_mapping.csv 불러오기
mapping_df = pd.read_csv("manual_mapping.csv", encoding='utf-8-sig')

# 딕셔너리로 변환
manual_mapping = dict(zip(mapping_df['raw_name'], mapping_df['normalized_name']))

def normalize_applicant(name):
    if pd.isna(name):
        return "UNKNOWN"
    name_std = str(name).strip()
    return manual_mapping.get(name_std, name_std)

# family_country에서 숫자를 추출하여 합산하는 함수
def extract_family_country_count(text):
    if pd.isna(text) or text == 0:
        return 0
    # 괄호 속 숫자만 추출하여 합산
    numbers = re.findall(r'\((\d+)\)', str(text))
    return sum(map(int, numbers))

# CSV 불러오기
df = pd.read_csv(INPUT_FILE_NAME, encoding='cp949')

# NaN 수치 데이터는 0으로 처리
df['cited_num'] = df['cited_num'].fillna(0)
df['family_country'] = df['family_country'].fillna(0)
df['claim_num'] = df['claim_num'].fillna(0)
df['referees_num'] = df['referees_num'].fillna(0)

# 출원인 표준화
df['normalized_applicant'] = df['applicant'].apply(normalize_applicant)

# 등록된 특허 필터링
df['is_registered'] = df['registration_num'].notnull()
reg_df = df[df['is_registered']].copy()

# family_country를 숫자형으로 변환한 새로운 열 생성
reg_df['family_country_count'] = reg_df['family_country'].apply(extract_family_country_count)

# 그룹별 계산용 시리즈 생성
group = reg_df.groupby('normalized_applicant')
patent_count = group.size()
citation_sum = group['cited_num'].sum()
cpp = citation_sum / patent_count
cpp.name = 'CPP'

# 평균 CPP 및 PII, TS 계산
avg_cpp = cpp.mean()
pii = (cpp / avg_cpp).fillna(0)
pii.name = 'PII'
ts = pii * patent_count
ts.name = 'TS'

# PFS 계산
avg_family_country_all = reg_df['family_country_count'].mean()
avg_family_country_by_applicant = group['family_country_count'].mean()
pfs = (avg_family_country_by_applicant / avg_family_country_all).fillna(0)
pfs.name = 'PFS'

# CRn, HHI 계산
total_patents = len(reg_df)
share = (patent_count / total_patents).fillna(0)
crn = share.sort_values(ascending=False).head(5).sum()
hhi = (share ** 2).sum()

# 통계 통합
stats_df = pd.concat([cpp, pii, ts, pfs], axis=1).reset_index()
stats_df.rename(columns={"normalized_applicant": "norm_applicant"}, inplace=True)

# 원본에 통계 병합
df = df.merge(stats_df, left_on='normalized_applicant', right_on='norm_applicant', how='left')
df.drop(columns=['norm_applicant'], inplace=True)

# 전체 지표 추가
df['CRn_5'] = crn
df['HHI'] = hhi

# 결과 저장
#df.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
OUTPUT_FILE_NAME = "output_with_metrics.xlsx"
df.to_excel(OUTPUT_FILE_NAME, index=False)

print(f"✅ 계산 완료! '{OUTPUT_FILE_NAME}' 파일로 저장되었습니다.")
