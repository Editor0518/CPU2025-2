import pandas as pd
import numpy as np
import os
import json
from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer

print("키프리스 특허 데이터 수집 및 토크나이징 프로그램")

# KorPatBERT 토크나이저 초기화
vocab_path = "./pretrained/korpat_vocab.txt"
if not os.path.exists(vocab_path):
    print(f"경고: {vocab_path} 파일이 없습니다.")
    exit(1)

tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)
print("[OK] KorPatBERT 토크나이저 초기화 완료")

# 사용자 입력
search_word = input("검색식을 입력하세요: ").strip()
withdrawn_input = input("취하/소멸/포기 포함 여부 (y/n): ").strip().lower()
withdrawn_flag = withdrawn_input == 'y'

print(f"검색 중... (검색어: {search_word})")
df = fetch_patents(search_word=search_word, year_to_search="0", withdrawn=withdrawn_flag)

if df.empty:
    print("검색 결과가 없습니다.")
    exit(0)

print(f"[OK] {len(df)}건의 특허 데이터 수집 완료")

max_len = 256

def process_text(text):
    """텍스트를 토크나이징"""
    if pd.isna(text) or text.strip() == "":
        return [0]*max_len, [0]*max_len, []
    
    # 토크나이징
    input_ids, segment_ids = tokenizer.encode(text, max_len=max_len)
    tokens = tokenizer.tokenize(text)
    
    return input_ids, segment_ids, tokens

print("토크나이징 진행 중...")
results = df['astrtCont'].fillna('').apply(process_text)

df['input_ids'] = results.apply(lambda x: x[0])
df['segment_ids'] = results.apply(lambda x: x[1])
df['tokens'] = results.apply(lambda x: x[2])

output_file = "patents_tokenized.pkl"
df.to_pickle(output_file)
print(f"[OK] 토크나이징 완료")
print(f"[OK] 데이터 저장 완료: {output_file}")

# 통계
print(f"\n=== 처리 결과 ===")
print(f"총 특허 건수: {len(df)}")
if len(df) > 0:
    token_lengths = df['tokens'].apply(len)
    print(f"평균 토큰 수: {token_lengths.mean():.1f}")
    print(f"최대 토큰 수: {token_lengths.max()}")
    print(f"최소 토큰 수: {token_lengths.min()}")
    print(f"데이터 컬럼: {list(df.columns)}")

# 샘플 출력
if len(df) > 0:
    print(f"\n=== 첫 번째 특허 샘플 ===")
    first_patent = df.iloc[0]
    title = str(first_patent.get('inventionTitle', 'N/A'))[:50]
    content = str(first_patent.get('astrtCont', ''))
    print(f"발명명: {title}...")
    print(f"원문 길이: {len(content)}")
    tokens = first_patent['tokens']
    print(f"토큰 수: {len(tokens)}")
    if len(tokens) > 0:
        sample_tokens = tokens[:10] if len(tokens) > 10 else tokens
        print(f"토큰 예시: {sample_tokens}")
    
    input_ids = first_patent['input_ids']
    print(f"input_ids 예시: {input_ids[:10]}")

print(f"\n프로그램 완료!")
print(f"저장된 파일: {output_file}")
print("토크나이징된 데이터는 BERT 모델 입력으로 사용할 수 있습니다.")
