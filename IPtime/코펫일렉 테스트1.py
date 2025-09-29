import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ElectraModel, AutoConfig
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import time

# 환경변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 설정값
ACCESS_TOKEN_PATH = "access_token.txt"
MODEL_FOLDER = "./electra"
REMOTE_MODEL_NAME = "KIPI-ai/KorPatElectra"
DATA_PATH = "patent_data_balanced.csv"
MAX_SEQ_LEN = 192
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 액세스 토큰
with open(ACCESS_TOKEN_PATH, "r") as f:
    access_token = f.read().strip()

# 모델 로딩 함수
def load_tokenizer_and_model():
    if os.path.exists(MODEL_FOLDER):
        print("🔍 로컬 ELECTRA 모델 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
        config = AutoConfig.from_pretrained(MODEL_FOLDER)
        base_model = ElectraModel.from_pretrained(MODEL_FOLDER, config=config)
    else:
        print("🌐 Hugging Face에서 ELECTRA 모델 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(REMOTE_MODEL_NAME, use_auth_token=access_token)
        config = AutoConfig.from_pretrained(REMOTE_MODEL_NAME, use_auth_token=access_token)
        base_model = ElectraModel.from_pretrained(REMOTE_MODEL_NAME, config=config, use_auth_token=access_token)
    return tokenizer, base_model, config

tokenizer, base_model, config = load_tokenizer_and_model()

# ===== 데이터 불러오기 =====
df = pd.read_csv(DATA_PATH)
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')

# ===== 사용자 정의 라벨 =====
label_list = ['AAA','AAB','AAC','ABA','ABB','ABC','ACA','ACB','ACC','ADA','ADB','ADC','ADD','AEA','AEB','AEC','AED','N']

# ===== 라벨 분포 분석 =====
print("\n📊 전체 데이터 라벨 분포 분석")
label_counts = df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    print(f"  {label}: {count}개 ({(count / len(df)) * 100:.1f}%)")

# ===== 데이터 필터링 =====
insufficient_labels = label_counts[label_counts < 2].index.tolist()
sufficient_labels = label_counts[label_counts >= 2].index.tolist()

if insufficient_labels:
    print(f"\n❌ 제외할 라벨: {insufficient_labels}")
    df = df[df['label'].isin(sufficient_labels)].copy()

# 원본 데이터에서 라벨별 개수 2개 이상만 유지
label_counts = df['label'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df_filtered = df[df['label'].isin(valid_labels)].copy()

# 라벨 인코딩 (필터링 후 한번만)
labels = sorted(df_filtered['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df_filtered['label_id'] = df_filtered['label'].map(label2id)

# 🔹 1차 분할: train vs temp (val+test)
train_df, temp_df = train_test_split(
    df_filtered,
    test_size=0.2,
    stratify=df_filtered['label_id'],
    random_state=42
)

# temp에서 다시 2개 미만 라벨 제거 (stratify 안전)
label_counts_temp = temp_df['label'].value_counts()
valid_labels_temp = label_counts_temp[label_counts_temp >= 2].index
temp_df = temp_df[temp_df['label'].isin(valid_labels_temp)].copy()
temp_df['label_id'] = temp_df['label'].map(label2id)


min_count = temp_df['label_id'].value_counts().min()
assert min_count >= 2, "❌ stratify를 위해 각 라벨이 최소 2개 이상이어야 합니다."

# 🔹 2차 분할: val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label_id'],
    random_state=42
)


# ===== 데이터셋 클래스 =====
class PatentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_SEQ_LEN):
        self.texts = df['text'].tolist()
        self.labels = df['label_id'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_loader = DataLoader(PatentDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PatentDataset(val_df, tokenizer), batch_size=BATCH_SIZE)
test_loader = DataLoader(PatentDataset(test_df, tokenizer), batch_size=BATCH_SIZE)

# ===== 분류 모델 정의 =====
class ElectraClassifier(nn.Module):
    def __init__(self, electra_model, num_labels):
        super().__init__()
        self.electra = electra_model
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_vector)

model = ElectraClassifier(base_model, len(label2id)).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    step_times = []

    progress_bar = tqdm(dataloader, desc="Training", ncols=100)
    for batch in progress_bar:
        start = time.time()
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        step_time = time.time() - start
        step_times.append(step_time)

        speed = 1 / step_time if step_time > 0 else 0
        progress_bar.set_postfix({
            "loss": f"{total_loss / (total / labels.size(0)):.4f}",
            "accuracy": f"{correct / total:.4f}",
            "it/s": f"{speed:.1f}"
        })

    avg_step_time = sum(step_times) / len(step_times)
    return total_loss / len(dataloader), correct / total, avg_step_time


def eval_epoch(model, dataloader, criterion, phase="Validation"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    step_times = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=phase, ncols=100)
        for batch in progress_bar:
            start = time.time()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            batch_loss = loss.item()
            total_loss += batch_loss
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            step_time = time.time() - start
            step_times.append(step_time)

            speed = 1 / step_time if step_time > 0 else 0
            progress_bar.set_postfix({
                "loss": f"{total_loss / (total / labels.size(0)):.4f}",
                "accuracy": f"{correct / total:.4f}",
                "it/s": f"{speed:.1f}"
            })

    avg_step_time = sum(step_times) / len(step_times)
    return total_loss / len(dataloader), correct / total, avg_step_time

# ===== 학습 루프 =====
print("_________________________________________________________________")
print("Total params:", sum(p.numel() for p in model.parameters()))
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Non-trainable params:", sum(p.numel() for p in model.parameters() if not p.requires_grad))
print("_________________________________________________________________")

best_val_loss = float('inf')
best_val_acc = 0
patience, early_stop_count = 3, 0

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_loss, train_acc, train_step_time = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_step_time = eval_epoch(model, val_loader, criterion)
    epoch_time = time.time() - start_time


    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"{len(train_loader)}/{len(train_loader)} [==============================] - "
          f"{int(epoch_time)}s {int(train_step_time)}s/step - accuracy: {train_acc:.4f} - loss: {train_loss:.4f} - "
          f"val_accuracy: {val_acc:.4f} - val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        early_stop_count = 0
        torch.save(model.state_dict(), "korpat_electra_classifier.pth")
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

# ===== 평가 =====
model.load_state_dict(torch.load("korpat_electra_classifier100.pth", weights_only=True))
test_loss, test_acc, _ = eval_epoch(model, test_loader, criterion, phase="Test")
print(f"\n📊 평가 결과: Accuracy={test_acc:.4f}, Loss={test_loss:.4f}")
