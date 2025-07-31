# korpatbert_classifier.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 메시지 숨기기

import pandas as pd
import numpy as np
from tqdm import tqdm
from korpat_tokenizer import Tokenizer  # PDF 제공 파일
import tensorflow as tf
from tensorflow import keras
#from keras import Dense
from tensorflow.keras.callbacks import EarlyStopping
import bert  # pip install bert-for-tf2


# ===== 경로 및 설정 =====
config_path = "./pretrained/korpat_bert_config.json"
vocab_path = "./pretrained/korpat_vocab.txt"
checkpoint_path = "./pretrained/model.ckpt-381250"

csv_path = "patent_data_finalN.csv"

MAX_SEQ_LEN = 192
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5

# ===== KorPat Tokenizer 선언 =====
tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)

# ===== 데이터 불러오기 =====
df = pd.read_csv(csv_path, encoding="cp949")
df = df[df['label'].notnull()].copy()
df["text"] = df["title"].fillna('') + " " + df["korean_summary"].fillna('')


# ===== 사용자 정의 라벨 =====
label_list = [
    'AAA', 'AAB', 'AAC',
    'ABA', 'ABB', 'ABC',
    'ACA', 'ACB', 'ACC',
    'ADA', 'ADB', 'ADC', 'ADD',
    'AEA', 'AEB', 'AEC', 'AED',
    'N'  # 노이즈
]

# ===== 라벨 분포 분석 =====
print("📊 전체 데이터 라벨 분포 분석")
print("=" * 50)

# 전체 라벨 분포 확인
label_counts = df['label'].value_counts().sort_index()
print(f"총 데이터 수: {len(df)}")
print(f"고유 라벨 수: {len(label_counts)}")
print("\n라벨별 데이터 수:")
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {label}: {count}개 ({percentage:.1f}%)")

# ===== 데이터 필터링 (최소 2개 이상 라벨만 유지) =====
print("\n🔍 라벨 필터링 (최소 2개 이상)")
print("=" * 30)

# 2개 미만인 라벨 찾기
insufficient_labels = label_counts[label_counts < 2].index.tolist()
sufficient_labels = label_counts[label_counts >= 2].index.tolist()

if insufficient_labels:
    print(f"❌ 제외할 라벨 (1개): {insufficient_labels}")
    print(f"✅ 사용할 라벨 ({len(sufficient_labels)}개): {sufficient_labels}")
    
    # 충분한 데이터가 있는 라벨만 필터링
    df_filtered = df[df['label'].isin(sufficient_labels)].copy()
    print(f"\n필터링 후 데이터 수: {len(df)} → {len(df_filtered)}")
else:
    print("✅ 모든 라벨이 2개 이상의 데이터를 가지고 있습니다.")
    df_filtered = df.copy()

# ===== 최종 라벨 맵핑 =====
# 필터링된 데이터의 라벨만 사용
labels = sorted(df_filtered["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df_filtered["label_id"] = df_filtered["label"].map(label2id)

for i, label in enumerate(labels):
    count = len(df_filtered[df_filtered['label'] == label])

# ===== 데이터 분리 (이제 안전하게 stratify 사용 가능) =====
print(f"\n🔄 데이터 분리 시작...")
from sklearn.model_selection import train_test_split

try:
    train_df, test_df = train_test_split(
        df_filtered, 
        test_size=0.1, 
        stratify=df_filtered["label_id"], 
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.1, 
        stratify=train_df["label_id"], 
        random_state=42
    )
    
    print(f"✅ 데이터 분리 성공!")
    print(f"  Train: {len(train_df)}개")
    print(f"  Validation: {len(val_df)}개") 
    print(f"  Test: {len(test_df)}개")
    
    # 분리 후 각 세트의 라벨 분포 확인
    print(f"\n📊 분리 후 라벨 분포:")
    for label in labels:
        train_count = len(train_df[train_df['label'] == label])
        val_count = len(val_df[val_df['label'] == label])
        test_count = len(test_df[test_df['label'] == label])
        total = train_count + val_count + test_count
        print(f"  {label}: Train({train_count}) + Val({val_count}) + Test({test_count}) = {total}")
    
except ValueError as e:
    print(f"❌ 데이터 분리 실패: {e}")
    print("라벨별 최소 데이터 수를 다시 확인하세요.")
    exit(1)

print("\n" + "="*50)
print("데이터 준비 완료! 학습을 시작합니다...")

# ===== 데이터 전처리 =====
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
print("\n모델 정의 중...")
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
        metrics=['accuracy'],
    )

model.summary()


early_stop = EarlyStopping(
    monitor='val_loss',      # validation loss 기준으로 감시
    patience=2,              # 개선 안 되는 epoch 수 (ex: 2번 연속)
    restore_best_weights=True  # 가장 좋은 성능의 가중치 복원
)

# ===== 학습 시작 =====
print("\n📚 모델 학습 시작...")
try:
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )
except Exception as e:
    print("❌ 학습 중 오류 발생:", e)

# ===== 평가 및 저장 =====
model.save("korpatBERT_patent_model.h5")
eval_result = model.evaluate(test_x, test_y)
print("\n📊 평가 결과:")
print("Accuracy: %.4f" % eval_result[1])
