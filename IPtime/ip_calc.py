import pandas as pd


# HHI 계산 시 total_patents가 0이면 share_squared는 int가 되어 .sum() 불가 -> 조건 분기 수정

if total_patents > 0:
    share_squared = (patent_counts / total_patents) ** 2
    hhi = share_squared.sum() * 10000
else:
    hhi = 0

# 나머지 계산 동일
citations_per_applicant = registered_df.groupby('applicant')['cited_num'].sum()
cpp = citations_per_applicant / patent_counts

avg_cpp = cpp.mean() if not cpp.empty else 0
pii = cpp / avg_cpp if avg_cpp > 0 else 0
ts = pii * patent_counts

family_sum = registered_df.groupby('applicant')['family_country'].sum()
family_avg = family_sum / patent_counts
global_avg_family = registered_df['family_country'].mean() if not registered_df.empty else 0
pfs = family_avg / global_avg_family if global_avg_family > 0 else 0

summary_df = pd.DataFrame({
    'Patent Count': patent_counts,
    'CPP': cpp,
    'PII': pii,
    'TS': ts,
    'PFS': pfs
}).sort_values(by='Patent Count', ascending=False)

indicators = {
    'CR5 (%)': crn,
    'HHI': hhi
}

summary_df.head(), indicators