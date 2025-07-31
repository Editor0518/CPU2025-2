# korpatbert_classifier_mid.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from tqdm import tqdm
from korpat_tokenizer import Tokenizer
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import bert

# ===== 경로 및 설정 =====
config_path = "./pretrained/korpat_bert_config.json"
vocab_path = "./pretrained/korpat_vocab.txt"
checkpoint_path = "./pretrained/model.ckpt-381250"
csv_path = "patent_data_finalN.csv"

MAX_SEQ_LEN = 192
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5

tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

# ===== 데이터 불러오기 및 중분류로 변환 =====
df = pd.read_csv(csv_path, encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')
df["label_fine"] = df["label"]
df["label"] = df["label"].apply(lambda x: x[:2] if x != 'N' else 'N')

# ===== 라벨 분포 =====
label_counts = df["label"].value_counts().sort_index()
insufficient_labels = label_counts[label_counts < 2].index.tolist()
df_filtered = df[~df["label"].isin(insufficient_labels)].copy()

# ===== 라벨 매핑 =====
labels = sorted(df_filtered["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df_filtered["label_id"] = df_filtered["label"].map(label2id)

# ===== 데이터 분리 =====
train_df, test_df = train_test_split(df_filtered, test_size=0.1, stratify=df_filtered["label_id"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label_id"], random_state=42)

# ===== 인코딩 =====
def encode_dataset(dataset):
    x_data, y_data = [], []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["text"]
        label_id = row["label_id"]
        ids, _ = tokenizer.encode(text, max_len=MAX_SEQ_LEN)
        x_data.append(ids)
        onehot = [0] * len(label2id)
        onehot[label_id] = 1
        y_data.append(onehot)
    return np.array(x_data), np.array(y_data)

train_x, train_y = encode_dataset(train_df)
val_x, val_y = encode_dataset(val_df)
test_x, test_y = encode_dataset(test_df)

# ===== 모델 정의 =====
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    input_ids = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    bert_params = bert.params_from_pretrained_ckpt(os.path.dirname(checkpoint_path))
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
    bert_output = l_bert(input_ids)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    outputs = keras.layers.Dense(units=len(label2id), activation='softmax')(cls_out)

    model = keras.Model(inputs=input_ids, outputs=outputs)
    model.build(input_shape=(None, MAX_SEQ_LEN))
    bert.load_stock_weights(l_bert, checkpoint_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# ===== 조기 종료 콜백 =====
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# ===== 학습 =====
history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# ===== 저장 및 평가 =====
model.save("korpatBERT_mid_model.h5")
eval_result = model.evaluate(test_x, test_y)
print("\n📊 평가 결과:")
print("Accuracy: %.4f" % eval_result[1])
