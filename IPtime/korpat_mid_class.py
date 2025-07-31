# korpatbert_classifier.py (중분류 기반)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from tqdm import tqdm
from korpat_tokenizer import Tokenizer
import tensorflow as tf
from tensorflow import keras
import bert
from sklearn.model_selection import train_test_split

# ===== GPU 설정 =====
tf.config.set_visible_devices([], 'GPU')

# ===== 경로 및 설정 =====
config_path = "./pretrained/korpat_bert_config.json"
vocab_path = "./pretrained/korpat_vocab.txt"
checkpoint_path = "./pretrained/model.ckpt-381250"
csv_path = "patent_data_finalN.csv"

MAX_SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-5

# ===== Tokenizer =====
tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

# ===== 데이터 불러오기 =====
df = pd.read_csv(csv_path, encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')

# ===== 라벨 처리: 중분류(label_mid)와 소분류(label_fine) =====
df['label_fine'] = df['label']  # 소분류 백업용 (ex: AAA)
df['label'] = df['label'].apply(lambda x: x[:2] if x != 'N' else 'N')  # 중분류로 변환

# ===== 라벨 분포 분석 =====
print("\n📊 전체 데이터 중분류 라벨 분포 분석")
label_counts = df['label'].value_counts().sort_index()
print(f"총 데이터 수: {len(df)}")
print(f"중분류 라벨 수: {len(label_counts)}")
for label, count in label_counts.items():
    print(f"  {label}: {count}개 ({(count/len(df))*100:.1f}%)")

# ===== 필터링 (2개 이상 데이터 유지) =====
insufficient_labels = label_counts[label_counts < 2].index.tolist()
sufficient_labels = label_counts[label_counts >= 2].index.tolist()

if insufficient_labels:
    print(f"\n❌ 제외할 라벨: {insufficient_labels}")
    df_filtered = df[df['label'].isin(sufficient_labels)].copy()
else:
    df_filtered = df.copy()

# ===== 라벨 매핑 =====
labels = sorted(df_filtered["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df_filtered["label_id"] = df_filtered["label"].map(label2id)

print("\n📋 최종 라벨 맵핑:")
for label, i in label2id.items():
    count = len(df_filtered[df_filtered['label'] == label])
    print(f"  {i}: {label} ({count}개)")

# ===== 데이터 분리 =====
train_df, test_df = train_test_split(df_filtered, test_size=0.1, stratify=df_filtered['label_id'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label_id'], random_state=42)

# ===== 인코딩 함수 =====
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

# ===== 모델 정의 및 학습 =====
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
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# ===== 학습 =====
history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# ===== 평가 및 저장 =====
model.save("korpatBERT_midlevel_model.h5")
eval_result = model.evaluate(test_x, test_y)
print("\n📊 평가 결과:")
print("Accuracy: %.4f" % eval_result[1])

# ===== [소분류 기반으로 학습할 경우를 대비한 참고용 코드 주석] =====
# df["label"] = df["label_fine"]  # 소분류 기준으로 되돌리기
# label_list = sorted(df["label"].unique())
# label2id = {label: i for i, label in enumerate(label_list)}
# df["label_id"] = df["label"].map(label2id)
