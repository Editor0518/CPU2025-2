# korpatbert_binary_classifier.py
# pip install tensorflow==2.6
# pip install keras==2.6
# pip install bert-for-tf2==0.14.9

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 숨기기
# tf.config.set_visible_devices([], 'GPU')  # GPU 강제 비활성화 → 주석 처리로 GPU 사용 가능

import pandas as pd
import numpy as np
from tqdm import tqdm
from korpat_tokenizer import Tokenizer
import tensorflow as tf
from tensorflow import keras
import bert
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# ===== 경로 및 설정 =====
config_path = "./pretrained/korpat_bert_config.json"
vocab_path = "./pretrained/korpat_vocab.txt"
checkpoint_path = "./pretrained/model.ckpt-381250"
csv_path = "modelTraining_noise.csv"

MAX_SEQ_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-5

# ===== Tokenizer 초기화 =====
tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

# ===== 데이터 불러오기 =====
df = pd.read_csv(csv_path, encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')

# ===== 이진 분류 라벨 생성 =====
df["binary_label"] = df["label"].apply(lambda x: 1 if x == 'N' else 0)
df["label_id"] = df["binary_label"]

print("📊 전체 데이터 수:", len(df))
print("  - N (노이즈):", (df["label_id"] == 1).sum())
print("  - 비N (정상):", (df["label_id"] == 0).sum())

# ===== 데이터 분할 =====
train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label_id"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label_id"],
    random_state=42
)

print("\n✅ 데이터 분리 완료")
print(f"  Train: {len(train_df)}개")
print(f"  Validation: {len(val_df)}개")
print(f"  Test: {len(test_df)}개")

# ===== 데이터 인코딩 =====
def encode_dataset(dataset):
    x_data, y_data = [], []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["text"]
        label_id = row["label_id"]
        ids, _ = tokenizer.encode(text, max_len=MAX_SEQ_LEN)
        x_data.append(ids)
        onehot = [0, 0]
        onehot[label_id] = 1
        y_data.append(onehot)
    return np.array(x_data), np.array(y_data)

train_x, train_y = encode_dataset(train_df)
val_x, val_y = encode_dataset(val_df)
test_x, test_y = encode_dataset(test_df)

# ===== 클래스 가중치 계산 =====
y_integers = np.argmax(train_y, axis=1)
computed_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = {i: w for i, w in enumerate(computed_class_weights)}
print("\n⚖️ 클래스 가중치:", class_weight_dict)

# ===== 모델 정의 및 학습 =====
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    input_ids = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')

    bert_params = bert.params_from_pretrained_ckpt(os.path.dirname(checkpoint_path))
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
    bert_output = l_bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.3)(cls_out)  # Dropout 추가
    outputs = keras.layers.Dense(units=2, activation='softmax')(cls_out)

    model = keras.Model(inputs=input_ids, outputs=outputs)
    model.build(input_shape=(None, MAX_SEQ_LEN))
    bert.load_stock_weights(l_bert, checkpoint_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

model.summary()

# ===== 콜백 설정 (EarlyStopping) =====
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ===== 모델 학습 =====
history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# ===== 평가 및 저장 =====
model.save("korpatBERT_noise_classifier.h5")
eval_result = model.evaluate(test_x, test_y)
print("\n📊 평가 결과:")
print("Accuracy: %.4f" % eval_result[1])
