import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from korpat_tokenizer import Tokenizer
import bert
from bert import BertModelLayer 


# ===== 기본 경로 설정 =====
list_of_sub_labels = [
    'AAA', 'AAB', 'AAC',
    'ABA', 'ABB', 'ABC',
    'ACA', 'ACB', 'ACC',
    'ADA', 'ADB', 'ADC', 'ADD',
    'AEA', 'AEB', 'AEC', 'AED',
    'N'  # 노이즈
]

list_of_mid_labels= ['AA', 'AB', 'AC', 'AD', 'AE', 'N']

CONFIG = {
    "noise": {
        "model_path": "korpatBERT_noise_filter.h5",
        "label": ['A', 'N'], #A=유효특허, N=노이즈
        "task_type": "binary"
    },
    "sub": {  # 소분류
        "model_path": "korpatBERT_patent_model.h5",
        "label": list_of_sub_labels,  # 소분류 라벨 리스트
        "task_type": "multi"
    },
    "mid": {  # 중분류
        "model_path": "korpatBERT_mid_model.h5",
        "label": list_of_mid_labels,  # 중분류 라벨 리스트
        "task_type": "multi"
    },
}

MAX_SEQ_LEN = 192
VOCAB_PATH = "./pretrained/korpat_vocab.txt"
tokenizer = Tokenizer(vocab_path=VOCAB_PATH, cased=True)

# ===== 텍스트 인코딩 함수 =====
def encode_texts(texts):
    input_ids = []
    for text in tqdm(texts):
        ids, _ = tokenizer.encode(text, max_len=MAX_SEQ_LEN)
        input_ids.append(ids)
    return np.array(input_ids)

# ===== 예측 함수 =====
def predict_labels(model, x_data, task_type, label_list):
    preds = model.predict(x_data)
    
    pred_labels = []
    pred_probs = []

    for p in preds:
        idx = np.argmax(p)
        if idx < len(label_list):
            pred_labels.append(label_list[idx])
        else:
            pred_labels.append("UNKNOWN")  # or skip
        pred_probs.append(float(np.max(p)))  # 최대 확률 저장

    return pred_labels, pred_probs



# ===== 메인 흐름 =====
def main():
    print("\n📌 예측 종류를 선택하세요:")
    print("1. 노이즈 분류")
    print("2. 소분류 분류")
    print("3. 중분류 분류")
    choice = input("선택 (1/2/3): ").strip()

    if choice == "1":
        mode = "noise"
    elif choice == "2":
        mode = "sub"
    elif choice == "3":
        mode = "mid"
    else:
        print("❌ 잘못된 선택입니다.")
        return

    # ===== CSV 입력 =====
    file_path = input("\n📂 예측할 CSV 파일 경로를 입력하세요: ").strip()
    if not os.path.exists(file_path):
        print("❌ 파일이 존재하지 않습니다.")
        return

    df = pd.read_csv(file_path, encoding="cp949")
    df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')
    x_data = encode_texts(df["text"].tolist())

    # ===== 모델 로드 =====
    print("📦 모델 로딩 중...")

    model = keras.models.load_model(CONFIG[mode]["model_path"], compile=False,
                                custom_objects={"BertModelLayer": BertModelLayer})

    labels = CONFIG[mode]["label"]
    task_type = CONFIG[mode]["task_type"]

        # ===== 예측 =====
    print("🔍 예측 중...")
    pred_labels, pred_probs = predict_labels(model, x_data, task_type, labels)
    df["label"] = pred_labels
    df["probability"] = pred_probs

    # ===== 결과 저장 =====
    output_path = f"prediction_result_{mode}.xlsx"
    df.to_excel(output_path, index=False, encoding='utf-8-sig', engine='openpyxl')
    print(f"\n✅ 예측 완료! 결과 파일 저장됨: {output_path}")


if __name__ == "__main__":
    main()
