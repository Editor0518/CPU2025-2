import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib


# 데이터 정의
data = {
    '출원인': [
        'SAMSUNG ELECTRONICS', 'TENCENT', 'LG ELECTRONICS',
        'BAIDU', 'ADOBE', 'MICROSOFT', 'GOOGLE', 'NVIDIA'
    ],
    'AAA': [18, 7, 4, 9, 0, 10, 4, 0],
    'AAB': [4, 4, 0, 1, 0, 1, 0, 1],
    'AAC': [2, 3, 2, 7, 0, 2, 0, 1],
    'ABA': [30, 11, 6, 6, 19, 5, 1, 0],
    'ABB': [5, 3, 0, 1, 5, 0, 0, 0],
    'ABC': [9, 3, 0, 1, 2, 0, 0, 0],
    'ACA': [5, 2, 0, 0, 0, 2, 2, 2],
    'ACB': [0, 1, 0, 0, 0, 0, 0, 1],
    'ACC': [6, 0, 0, 0, 0, 0, 0, 0],
    'ADA': [16, 1, 22, 4, 0, 1, 1, 5],
    'ADB': [0, 0, 0, 0, 0, 0, 0, 0],
    'ADC': [0, 1, 0, 0, 0, 0, 0, 0],
    'ADD': [0, 1, 2, 0, 0, 0, 0, 0],
    'AEA': [3, 3, 0, 1, 1, 3, 2, 0],
    'AEB': [2, 1, 0, 0, 0, 0, 0, 0],
    'AEC': [1, 1, 0, 2, 0, 3, 1, 1],
    'AED': [3, 0, 2, 3, 1, 1, 0, 0]
}
data = {
    "출원인": [
        "SAMSUNG ELECTRONICS", "TENCENT", "LG ELECTRONICS",
        "BAIDU", "ADOBE", "MICROSOFT", "GOOGLE", "NVIDIA"
    ],
    "AA": [24, 14, 6, 17, 0, 13, 4, 2],
    "AB": [44, 17, 6, 8, 26, 5, 1, 0],
    "AC": [11, 3, 0, 0, 0, 2, 2, 3],
    "AD": [16, 3, 24, 4, 0, 1, 1, 5],
    "AE": [9, 5, 2, 6, 2, 7, 3, 1]
}

# 데이터프레임 생성
df = pd.DataFrame(data)
df.set_index('출원인', inplace=True)

# 시각화
plt.figure(figsize=(18, 6))  # 너비 넓게
sns.set(font_scale=0.9)

ax = sns.heatmap(
    df,
    annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5,
    cbar_kws={'label': '출원 건수'}
)

plt.title("출원인별 기술분류별 특허 출원 수 (히트맵)", fontsize=14)
plt.xlabel("기술 분류")
plt.ylabel("출원인")
plt.tight_layout()
plt.show()
