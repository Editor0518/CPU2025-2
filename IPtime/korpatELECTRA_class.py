import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ===== 설정 =====
ACCESS_TOKEN_PATH = "access_token.txt"
MODEL_NAME = "KIPI‑ai/KorPatElectra"
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 3e-5

# ===== ACCESS TOKEN 로딩 함수 =====
def load_token_from_file(path=ACCESS_TOKEN_PATH):
    with open(path, "r") as f:
        return f.read().strip()

ACCESS_TOKEN = load_token_from_file()

# ===== 모델 및 토크나이저 로딩 =====
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME, use_auth_token=ACCESS_TOKEN)
base_model = ElectraModel.from_pretrained(MODEL_NAME, use_auth_token=ACCESS_TOKEN)

# ===== 데이터 불러오기 =====
df = pd.read_csv("modelTraining3.csv", encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna("") + " " + df["korean_summary"].fillna("")

label_list = [
    'AAA', 'AAB', 'AAC',
    'ABA', 'ABB', 'ABC',
    'ACA', 'ACB', 'ACC',
    'ADA', 'ADB', 'ADC', 'ADD',
    'AEA', 'AEB', 'AEC', 'AED',
    'N'
]
df = df[df['label'].isin(label_list)].copy()
label2id = {label:i for i,label in enumerate(sorted(df['label'].unique()))}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.1, stratify=df["label_id"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label_id"], random_state=42)
test_df = temp_df

# ===== Dataset 정의 =====
class PatentDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx],
                        max_length=MAX_SEQ_LEN,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx])

# ===== DataLoader =====
train_loader = DataLoader(PatentDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PatentDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(PatentDataset(test_df), batch_size=BATCH_SIZE)

# ===== 분류 모델 정의 =====
class ElectraClassifier(nn.Module):
    def __init__(self, electra_model, num_labels):
        super().__init__()
        self.electra = electra_model
        self.classifier = nn.Linear(self.electra.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ElectraClassifier(base_model, len(label2id)).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ===== Training & Validation Loop =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch, labels in train_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        labels = labels.to(device)
        optim.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            labels = labels.to(device)
            logits = model(**batch)
            total_val_loss += criterion(logits, labels).item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
    val_acc = correct / len(val_df)
    print(f"Epoch {epoch+1}/{EPOCHS} — Train loss: {avg_train_loss:.4f}, Val loss: {total_val_loss/len(val_loader):.4f}, Val acc: {val_acc:.4f}")

# ===== 테스트 평가 =====
model.eval()
correct = 0
with torch.no_grad():
    for batch, labels in test_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        labels = labels.to(device)
        logits = model(**batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
test_acc = correct / len(test_df)
print(f"Test accuracy: {test_acc:.4f}")

# Optional: 모델 저장
torch.save(model.state_dict(), "korpat_electra_classifier.pt")
