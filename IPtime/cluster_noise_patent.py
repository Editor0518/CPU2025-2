import pandas as pd
from collections import defaultdict

# 입력/출력 파일
INPUT_CSV = "noises.csv"
OUTPUT_XLSX = "clustered_noise_patents.xlsx"

# 분석 대상 텍스트 열
TEXT_COLS = ['title', 'korean_summary', 'summary', 'main_claim']

# 데이터 불러오기
df = pd.read_csv(INPUT_CSV, encoding='cp949')
df.fillna("", inplace=True)
df['merged_text'] = df[TEXT_COLS].agg(' '.join, axis=1).str.lower()

# 클러스터링 결과 저장
cluster_result = defaultdict(list)
cluster_count = defaultdict(int)
assigned_ids = set()

# 중분류 우선순위 및 소분류 목록
CLUSTER_SIDS = {
    "AE": ["AEA", "AEB", "AEC", "AED"],
    "AD": ["ADA", "ADB", "ADC", "ADD"],
    "AC": ["ACA", "ACB", "ACC"],
    "AB": ["ABA", "ABB", "ABC"],
    "AA": ["AAA", "AAB", "AAC"]
}

# 텍스트 조건 판별 함수
def match_condition(text, sid):
    if sid == "AEB":
        return ("테스트" in text or "훈련용" in text) and "생성" in text
    elif sid == "ADD":
        group1 = ("잡음" in text or "노이즈" in text) and any(w in text for w in ["음성", "음악", "소리"]) and "제거" in text
        group2 = "복원" in text and any(w in text for w in ["음성", "음악", "소리"])
        return group1 or group2
    elif sid == "ACA":
        return "동영상" in text and not ("합성" in text or "복원" in text)
    elif sid == "ACB":
        return "동영상" in text and "합성" in text
    elif sid == "ACC":
        return "동영상" in text and "복원" in text
    elif sid == "ABB":
        return any(w in text for w in ["이미지", "영상"]) and any(w in text for w in ["변형", "보정", "해상도", "화질", "선명"])
    elif sid == "ABC":
        cond1 = any(w in text for w in ["이미지", "영상"]) and any(w in text for w in ["합성", "combine", "composite"])
        cond2 = "이미지" in text and any(w in text for w in ["텍스트", "자막", "타이틀", "문구"]) and "생성" in text
        return cond1 or cond2
    elif sid == "ABA":
        return any(w in text for w in ["이미지", "영상"]) and "생성" in text
    elif sid == "AAA":
        return any(w in text for w in ["대화", "문장", "문서", "보고서"]) and "생성" in text
    elif sid == "AAB":
        return any(w in text for w in ["번역", "translation"])
    elif sid == "AAC":
        cond1 = any(w in text for w in ["텍스트", "내용"]) and \
                any(w in text for w in ["요약", "분석"]) and \
                "생성" in text
        cond2 = any(w in text for w in ["텍스트", "내용"]) and \
                any(w in text for w in ["결과", "솔루션"]) and \
                any(w in text for w in ["전달", "전송"])
        return cond1 or cond2
    elif sid == "AEA":
        return "멀티모달" in text or "멀티 모달" in text
    elif sid == "AEC":
        return any(w in text for w in ["3d", "3차원", "다각도"])
    elif sid == "AED":
        return "시나리오" in text
    elif sid == "ADA":
        return any(w in text for w in ["음성", "voice"])
    elif sid == "ADB":
        return any(w in text for w in ["음악", "music"])
    elif sid == "ADC":
        return any(w in text for w in ["소리", "environmental sound", "환경음"])
    else:
        return False

# 클러스터링 수행
for mid in ["AE", "AD", "AC", "AB", "AA"]:
    for sid in CLUSTER_SIDS[mid]:
        mask = df['merged_text'].apply(lambda text: match_condition(text, sid)) & (~df['id'].isin(assigned_ids))
        matched = df[mask]
        if not matched.empty:
            cluster_result[sid].extend(matched.to_dict('records'))
            cluster_count[sid] += len(matched)
            assigned_ids.update(matched['id'])

# 미분류 처리
unclassified_df = df[~df['id'].isin(assigned_ids)]
unclassified_count = len(unclassified_df)

# 엑셀로 저장
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    for sid, records in cluster_result.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=sid, index=False)
    if unclassified_count > 0:
        unclassified_df.to_excel(writer, sheet_name="Unclassified", index=False)

# 출력
summary = defaultdict(int)
print("📊 소분류별 클러스터링 개수:")
for sid, count in cluster_count.items():
    print(f"  {sid}: {count}개")
    summary[sid[:2]] += count

print("\n📂 중분류별 요약:")
for mid, count in summary.items():
    print(f"  {mid}: {count}개")

total_clustered = sum(cluster_count.values())
print(f"\n✅ 클러스터링된 총 특허 수: {total_clustered}개")
print(f"❗ 클러스터링되지 않은 특허 수: {unclassified_count}개")
