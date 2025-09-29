import pandas as pd

# --- 비교 필요한 모델에 따라 주석 해제 후 사용하기 ---
# KorPatBERT 예측 결과 파일 경로 설정
PREDICTION_SUBCLASS_FILE = "prediction_result_sub2.xlsx"
PREDICTION_NOISE_FILE = "prediction_result_noise.xlsx"
PREDICTION_MIDCLASS_FILE = "prediction_result_mid.xlsx"

# KorPatELECTRA 예측 결과 파일 경로 설정
PREDICTION_SUBCLASS_FILE = "prediction_result_el_sub.xlsx"
#PREDICTION_NOISE_FILE = "prediction_result_el_noise.xlsx"
#PREDICTION_MIDCLASS_FILE = "prediction_result_el_mid.xlsx"
#----------------------------------------------------

ANSWER_FILE = "patent_data_finalN.csv"
ANSWER_FILE = "patent_data_balanced_deduplicated.csv"
PREDICTION_NOISE_FILE = "prediction_result_el_sub.xlsx"
PREDICTION_MIDCLASS_FILE = "prediction_result_el_sub.xlsx"

# ===== 정답 데이터 불러오기 =====
df_true = pd.read_csv(ANSWER_FILE)
df_true = df_true[df_true['label'].notnull()].copy().reset_index(drop=True)

# 공통 텍스트 기준 열 생성 (title + korean_summary)
df_true["text"] = df_true["title"].fillna("") + " " + df_true["korean_summary"].fillna("")

# 텍스트 기반 매칭을 위한 키 생성
def normalize_text(text):
    return str(text).strip().replace(" ", "")

df_true["match_key"] = df_true["text"].apply(normalize_text)

# 정답 데이터를 딕셔너리로 변환하여 빠른 비교 준비
true_labels = dict(zip(df_true["match_key"], df_true["label"]))

# ===== 예측 결과 불러오기 함수 =====
def load_prediction(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df["text"] = df["title"].fillna("") + " " + df["korean_summary"].fillna("")
    df["match_key"] = df["text"].apply(normalize_text)
    return df

# ===== 정확도 비교 함수: Sub (정확히 일치) =====
def evaluate_sub_loop(pred_df):
    correct = 0
    total = 0
    for idx, row in pred_df.iterrows():
        pred_key = row["match_key"]
        if pred_key in true_labels:
            total += 1
            pred_label = str(row["label"])
            true_label = str(true_labels[pred_key])
            if pred_label == true_label:
                correct += 1
    acc = correct / total
    print("\n📘 [소분류] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 정확도 비교 함수: Noise ('N'인지 여부만 비교) =====
def evaluate_noise_loop(pred_df):
    correct = 0
    total = 0
    for idx, row in pred_df.iterrows():
        pred_key = row["match_key"]
        if pred_key in true_labels:
            total += 1
            pred_label = str(row["label"])
            true_label = str(true_labels[pred_key])
            is_pred_noise = (pred_label == 'N')
            is_true_noise = (true_label == 'N')
            if is_pred_noise == is_true_noise:
                correct += 1
    acc = correct / total
    print("\n📕 [노이즈] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 정확도 비교 함수: Mid (N이면 N 비교, 아니면 앞 두자리 비교) =====
def evaluate_mid_loop(pred_df):
    correct = 0
    total = 0
    for idx, row in pred_df.iterrows():
        pred_key = row["match_key"]
        if pred_key in true_labels:
            total += 1
            pred_label = str(row["label"])
            true_label = str(true_labels[pred_key])
            if pred_label == 'N':
                if true_label == 'N':
                    correct += 1
            elif true_label.startswith(pred_label):
                 correct += 1
    acc = correct / total
    print("\n📙 [중분류] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 실행 =====
evaluate_noise_loop(load_prediction(PREDICTION_NOISE_FILE))
evaluate_sub_loop(load_prediction(PREDICTION_SUBCLASS_FILE))
evaluate_mid_loop(load_prediction(PREDICTION_MIDCLASS_FILE))