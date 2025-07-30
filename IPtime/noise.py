

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# ===================== 1. 데이터 로딩 =====================
df = pd.read_csv("modelTraining3.CSV", encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')

# ===================== 2. 라벨 인코딩 =====================
labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# ===================== 3. 데이터 분할 =====================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label_id"].tolist(),
    test_size=0.2, stratify=df["label_id"], random_state=42
)

# ===================== 4. 토크나이저 및 Dataset 정의 =====================
MODEL_NAME = "kipi-ai/KorPatBERT"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class PatentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = PatentDataset(train_texts, train_labels, tokenizer)
val_dataset = PatentDataset(val_texts, val_labels, tokenizer)

# ===================== 5. 모델 정의 및 학습 =====================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
