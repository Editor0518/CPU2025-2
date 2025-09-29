import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import ElectraModel

# ===== 기본 설정 =====
BATCH_SIZE = 8

list_of_sub_labels = ['ACB','AAC',
    'AAA','ABB', 'AAB', 
    'ABA',  'ABC',
    'ACA',
]

list_of_mid_labels = ['AA', 'AB', 'AC', 'AD', 'AE', 'N']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "noise": {
        "model_path": "korpat_electra_noise_classifier.pth",
        "label": ['non-N', 'N'],
        "task_type": "binary"
    },
    "sub": {
        "model_path": "korpat_electra_classifier.pth",
        "label": list_of_sub_labels,
        "task_type": "multi"
    },
    "mid": {
        "model_path": "korpat_electra_mid_classifier.pth",
        "label": list_of_mid_labels,
        "task_type": "multi"
    },
}

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_SEQ_LEN = 192
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)


# 추론 코드에서 ElectraClassifier 수정
class ElectraClassifier(nn.Module):
    def __init__(self, electra_model, num_labels):
        super().__init__()
        self.electra = electra_model
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_vector)


# ===== 텍스트 인코딩 =====
def encode_texts(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )

# ===== 예측 함수 =====
def predict_labels(model, inputs, label_list, batch_size=8, threshold=0.5):
    model.eval()
    model.to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    num_samples = input_ids.size(0)
    pred_labels = []
    pred_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs
            probs = torch.softmax(logits, dim=1)

            # 2진 분류인 경우 threshold 적용
            if len(label_list) == 2:
                # 'non-N'의 확률을 기준으로 예측
                preds = (probs[:, 0] >= threshold).long()
                # preds==1이면 non-N, 0이면 N. label_list의 순서에 맞게 변환
                # label_list가 ['non-N', 'N']이므로, preds=1이면 non-N(idx 0), preds=0이면 N(idx 1)
                preds = torch.where(probs[:, 0] >= threshold, torch.tensor(0), torch.tensor(1))
            else:
                max_probs, preds = torch.max(probs, dim=1)

            for pred_idx, prob in zip(preds.tolist(), probs.tolist()):
                pred_labels.append(label_list[pred_idx])
                pred_probs.append(prob[pred_idx])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    file_path = input("\n📂 예측할 CSV 파일 경로를 입력하세요: ").strip()
    if not os.path.exists(file_path):
        print("❌ 파일이 존재하지 않습니다.")
        return

    df = pd.read_csv(file_path)
    df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')
    texts = df["text"].tolist()

    print("📦 모델 로딩 중...")
    
    # === 수정된 부분: 선택에 따라 모델 경로와 레이블 수를 동적으로 설정 ===
    model_config = CONFIG[mode]
    num_labels = len(model_config["label"])
    model_path = model_config["model_path"]

    base_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = ElectraClassifier(base_model, num_labels=num_labels)
    
    # weight_only=True 옵션은 PyTorch 2.0.1 이상에서만 사용 가능합니다.
    # 버전이 낮다면, 이 부분을 삭제하거나 False로 설정하세요.
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()

    print("🧼 텍스트 인코딩 중...")
    inputs = encode_texts(texts)

    print("🔍 예측 중...")
    pred_labels, pred_probs = predict_labels(model, inputs, model_config["label"], batch_size=BATCH_SIZE)

    df["label"] = pred_labels
    df["probability"] = pred_probs

    # 🔹 통계 정보 출력
    total = len(df)
    print(f"\n📊 예측된 총 샘플 수: {total}")
    
    label_counts = df["label"].value_counts()
    print("📁 예측된 라벨별 개수:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count}개")

    avg_prob = df["probability"].mean()
    print(f"\n📈 평균 예측 확신도 (Confidence): {avg_prob:.4f}")

    output_path = f"prediction_result_el_{mode}.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n✅ 예측 완료! 결과 파일 저장됨: {output_path}")


if __name__ == "__main__":
    main()