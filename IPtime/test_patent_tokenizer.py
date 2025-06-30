# test_patent_tokenizer.py - patent_search.py의 토크나이저 부분 테스트
import pandas as pd
import os
from korpat_tokenizer import Tokenizer

print("KorPatBERT 토크나이저 테스트")

# KorPatBERT 토크나이저 초기화
vocab_path = "./pretrained/korpat_vocab.txt"
if not os.path.exists(vocab_path):
    print(f"경고: {vocab_path} 파일이 없습니다.")
    exit(1)

try:
    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)
    print("✔ KorPatBERT 토크나이저 초기화 완료")
except Exception as e:
    print(f"토크나이저 초기화 오류: {e}")
    exit(1)

# 테스트 데이터 생성
test_data = {
    'astrtCont': [
        "본 발명은 반도체 장치에 관한 것으로, 특히 메모리 소자의 성능을 향상시키는 기술에 관한 것이다.",
        "본 고안은 주로 일회용 합성세제액을 집어넣어 밀봉하는 세제액포에 관한 것이다.",
        "",  # 빈 문자열 테스트
        None  # None 값 테스트
    ],
    'inventionTitle': [
        "반도체 메모리 장치",
        "합성세제 액포",
        "빈 문서",
        "None 문서"
    ]
}

df = pd.DataFrame(test_data)
print(f"테스트 데이터 생성: {len(df)}건")

# KorPatBERT 토크나이징
max_len = 256

def tokenize_text(text):
    """텍스트를 토크나이징하여 input_ids, segment_ids 반환"""
    if pd.isna(text) or text.strip() == "":
        # 빈 텍스트의 경우 패딩으로 채움
        return [0] * max_len, [0] * max_len
    
    try:
        input_ids, segment_ids = tokenizer.encode(text, max_len=max_len)
        return input_ids, segment_ids
    except Exception as e:
        print(f"토크나이징 오류: {e}")
        # 오류 발생 시 패딩으로 채움
        return [0] * max_len, [0] * max_len

print("토크나이징 진행 중...")

# 초록(astrtCont) 토크나이징
tokenizing_results = df['astrtCont'].fillna('').apply(tokenize_text)
df['input_ids'] = tokenizing_results.apply(lambda x: x[0])
df['segment_ids'] = tokenizing_results.apply(lambda x: x[1])

# 추가로 토큰 정보도 저장 (디버깅용)
df['tokens'] = df['astrtCont'].fillna('').apply(lambda text: 
    tokenizer.tokenize(text) if text.strip() != "" else []
)

print("✔ 토크나이징 완료")

# 결과 출력
print(f"\n=== 처리 결과 ===")
for i, row in df.iterrows():
    print(f"\n{i+1}. {row['inventionTitle']}")
    print(f"   원문: {str(row['astrtCont'])[:50]}...")
    print(f"   토큰 수: {len(row['tokens'])}")
    print(f"   input_ids 앞 10개: {row['input_ids'][:10]}")
    print(f"   토큰 예시: {row['tokens'][:5]}")

print("\n✔ 토크나이저 테스트 완료!")
