import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

# CSV 파일 로드
df = pd.read_csv("patents_clustered_topics.csv")

# 🔹 클러스터 분포 시각화
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_counts.plot(kind='bar')
plt.title("클러스터별 특허 건수")
plt.xlabel("클러스터")
plt.ylabel("건수")
plt.show()

# 🔹 토픽 분포 시각화
topic_counts = df['topic_lda'].value_counts().sort_index()
topic_counts.plot(kind='bar', color='orange')
plt.title("토픽별 특허 건수")
plt.xlabel("토픽")
plt.ylabel("건수")
plt.show()

# 🔹 클러스터-토픽 조합 히트맵
pivot = df.pivot_table(index='topic_lda', columns='cluster', aggfunc='size', fill_value=0)
sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues')
plt.title("클러스터-토픽 히트맵")
plt.xlabel("클러스터")
plt.ylabel("토픽")
plt.show()
