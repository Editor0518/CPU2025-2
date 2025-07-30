# predict_korpatbert.py

import numpy as np
import pandas as pd
from korpat_tokenizer import Tokenizer
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# ===== 사전 설정 =====
MODEL_PATH = "korpatBERT_patent_model.h5"
VOCAB_PATH = "./pretrained/korpat_vocab.txt"
MAX_SEQ_LEN = 256

# ===== 사용자 정의 라벨 (AAA, AAB, ... + N)
label_list = [
    'AAA', 'AAB', 'AAC',
    'ABA', 'ABB', 'ABC',
    'ACA', 'ACB', 'ACC',
    'ADA', 'ADB', 'ADC', 'ADD',
    'AEA', 'AEB', 'AEC',
    'N'  # 노이즈
]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# ===== KorPat Tokenizer 로드 =====
tokenizer = Tokenizer(vocab_path=VOCAB_PATH, cased=True)

# ===== 모델 로드 =====
model = keras.models.load_model(MODEL_PATH)

# ===== 예측 함수 정의 =====
def predict_labels(texts):
    results = []
    for text in tqdm(texts, desc="예측 중"):
        ids, _ = tokenizer.encode(text, max_len=MAX_SEQ_LEN)
        input_arr = np.array([ids])
        pred = model.predict(input_arr, verbose=0)
        pred_id = int(np.argmax(pred))
        pred_label = id2label[pred_id]
        results.append(pred_label)
    return results

# ===== 입력 CSV 파일 예시 =====
df = pd.read_csv("input.csv", encoding="cp949")
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')

# ===== 예측 실행 =====
df["predicted_label"] = predict_labels(df["text"].tolist())

# ===== 출력 저장 =====
df.to_csv("predicted_output.csv", encoding="utf-8-sig", index=False)
print("✅ 예측 완료 → predicted_output.csv 저장됨")
