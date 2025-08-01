import pandas as pd

# CSV 파일 경로
file_path = "AS특허만있음.csv"  # ← 여기에 실제 파일명을 입력하세요

# CSV 파일 읽기
df = pd.read_csv(file_path, encoding="cp949")

# 'label' 열의 값 개수 세기
label_counts = df['label'].value_counts()

# 결과 출력
print(label_counts)

# 필요하면 DataFrame으로 저장해서 엑셀이나 CSV로 출력도 가능
label_counts.to_csv("label_counts.csv")
