# patent_search_with_embeddings.py - 임베딩 포함 버전
import pandas as pd
import numpy as np
import os
import json
from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer

print("키프리스 특허 데이터 수집, 토크나이징, 임베딩 프로그램")

# 먼저 토크나이징된 데이터가 있는지 확인
tokenized_file = "patents_tokenized.pkl"
if os.path.exists(tokenized_file):
    choice = input(f"{tokenized_file}가 존재합니다. 기존 데이터를 사용하시겠습니까? (y/n): ").strip().lower()
    if choice == 'y':
        df = pd.read_pickle(tokenized_file)
        print(f"[OK] 기존 토크나이징 데이터 로드: {len(df)}건")
    else:
        print("새로 데이터를 수집합니다.")
        exit(0)
else:
    print(f"{tokenized_file}가 없습니다. 먼저 patent_search.py를 실행하세요.")
    exit(1)

# BERT 라이브러리 확인
try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Transformers 라이브러리 사용 (더 안정적)
    from transformers import TFBertModel, BertTokenizer
    print("[OK] Transformers 라이브러리 사용")
    USE_TRANSFORMERS = True
    
except ImportError:
    print("Transformers 라이브러리가 없습니다. 설치하세요: pip install transformers")
    USE_TRANSFORMERS = False

if USE_TRANSFORMERS:
    try:
        # KorPatBERT 또는 유사한 한국어 BERT 모델 사용
        model_name = "monologg/kobert"  # 또는 다른 한국어 BERT 모델
        
        print(f"BERT 모델 로딩 중: {model_name}")
        bert_model = TFBertModel.from_pretrained(model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        
        print("[OK] BERT 모델 로딩 완료")
        
        def get_bert_embedding(text, max_length=256):
            """BERT 임베딩 생성"""
            try:
                # 텍스트 토크나이징
                inputs = bert_tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='tf'
                )
                
                # BERT 모델 통과
                outputs = bert_model(inputs)
                
                # [CLS] 토큰의 임베딩 사용 (첫 번째 토큰)
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                
                return cls_embedding
                
            except Exception as e:
                print(f"임베딩 생성 오류: {e}")
                return np.zeros(768)
        
        # 임베딩 생성
        print("BERT 임베딩 생성 중...")
        embeddings = []
        
        for i, text in enumerate(df['astrtCont'].fillna('')):
            if i % 50 == 0:
                print(f"진행률: {i}/{len(df)}")
            
            if text.strip() == "":
                embeddings.append(np.zeros(768))
            else:
                embedding = get_bert_embedding(text)
                embeddings.append(embedding)
        
        df['bert_embedding'] = embeddings
        
        # 저장
        output_file = "patents_tokenized_with_embeddings.pkl"
        df.to_pickle(output_file)
        
        print(f"[OK] 임베딩 완료!")
        print(f"[OK] 저장 완료: {output_file}")
        print(f"임베딩 차원: {embeddings[0].shape}")
        
    except Exception as e:
        print(f"BERT 임베딩 생성 실패: {e}")
        print("토크나이징 데이터만 사용하세요.")

else:
    print("임베딩 라이브러리가 없어 토크나이징 데이터만 사용 가능합니다.")
    print("임베딩이 필요하면 다음을 설치하세요:")
    print("pip install transformers torch")

print("\n프로그램 완료!")
