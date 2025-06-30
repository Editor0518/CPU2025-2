# patent_search.py
import pandas as pd
import os
from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer

print("키프리스 특허 데이터 수집 및 토크나이징 프로그램")

# KorPatBERT 토크나이저 초기화
vocab_path = "./pretrained/korpat_vocab.txt"
if not os.path.exists(vocab_path):
    print(f"경고: {vocab_path} 파일이 없습니다. 현재 디렉토리의 파일을 확인하세요.")
    # 다른 가능한 경로들 시도
    possible_paths = ["./korpat_vocab.txt", "../pretrained/korpat_vocab.txt", "./korpatbert_v1.0/korpat_vocab.txt"]
    for path in possible_paths:
        if os.path.exists(path):
            vocab_path = path
            print(f"발견된 vocab 파일: {vocab_path}")
            break
    else:
        print("vocab 파일을 찾을 수 없습니다. 기본 토크나이저로 실행합니다.")

try:
    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)
    print("[OK] KorPatBERT 토크나이저 초기화 완료")
except Exception as e:
    print(f"토크나이저 초기화 오류: {e}")
    exit(1)

# 사용자 입력
search_word = input("검색식을 입력하세요: ").strip()
withdrawn_input = input("취하/소멸/포기 포함 여부 (y/n): ").strip().lower()
withdrawn_flag = withdrawn_input == 'y'

print(f"검색 중... (검색어: {search_word})")

# 데이터 수집
df = fetch_patents(
    search_word=search_word,
    year_to_search="0",
    withdrawn=withdrawn_flag
)

if df.empty:
    print("검색 결과가 없습니다.")
    exit(0)

print(f"[OK] {len(df)}건의 특허 데이터 수집 완료")

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

print("[OK] 토크나이징 완료")

# 결과 저장
output_file = "patents_tokenized.pkl"
df.to_pickle(output_file)
print(f"[OK] tokenized 데이터 저장 완료: {output_file}")

# 간단한 통계 출력
print(f"\n=== 처리 결과 ===")
print(f"총 특허 건수: {len(df)}")
if len(df) > 0 and 'tokens' in df.columns:
    token_lengths = df['tokens'].apply(len)
    print(f"평균 토큰 수: {token_lengths.mean():.1f}")
    print(f"최대 토큰 수: {token_lengths.max()}")
    print(f"최소 토큰 수: {token_lengths.min()}")

# 샘플 출력 (첫 번째 특허)
if len(df) > 0:
    print(f"\n=== 첫 번째 특허 샘플 ===")
    first_patent = df.iloc[0]
    title = str(first_patent.get('inventionTitle', 'N/A'))[:50]
    content = str(first_patent.get('astrtCont', ''))
    print(f"발명명: {title}...")
    print(f"원문 길이: {len(content)}")
    if 'tokens' in df.columns:
        tokens = first_patent['tokens']
        print(f"토큰 수: {len(tokens)}")
        if len(tokens) > 0:
            sample_tokens = tokens[:10] if len(tokens) > 10 else tokens
            print(f"토큰 예시: {sample_tokens}")
    
    if 'input_ids' in df.columns:
        input_ids = first_patent['input_ids']
        print(f"input_ids 예시: {input_ids[:10]}")

print(f"\n프로그램 완료.")
print(f"저장된 파일: {output_file}")
print(f"데이터 컬럼: {list(df.columns)}")
