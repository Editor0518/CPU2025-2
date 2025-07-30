from transformers import BertConfig, load_tf_weights_in_bert
import tensorflow as tf
import torch
import os
import shutil

# === 설정 ===
TF_CHECKPOINT_PATH = "./pretrained/model.ckpt-381250"
BERT_CONFIG_FILE = "./pretrained/korpat_bert_config.json"
OUTPUT_DIR = "./korpatbert_hf"

# === 출력 디렉토리 준비 ===
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 1. BERT config 불러오기 ===
config = BertConfig.from_json_file(BERT_CONFIG_FILE)

# === 2. PyTorch 모델 생성 및 가중치 변환 ===
from transformers import BertModel

model = BertModel(config)
load_tf_weights_in_bert(model, config, TF_CHECKPOINT_PATH)

# === 3. 모델 저장 ===
model.save_pretrained(OUTPUT_DIR)

# === 4. vocab.txt 복사 ===
shutil.copyfile("./korpat_bert/korpat_vocab.txt", os.path.join(OUTPUT_DIR, "vocab.txt"))

print(f"✅ 변환 완료! Hugging Face 포맷으로 저장됨: {OUTPUT_DIR}")
