import pandas as pd

# --- 비교 필요한 모델에 따라 주석 해제 후 사용하기 ---
# KorPatBERT 예측 결과 파일 경로 설정
PREDICTION_SUBCLASS_FILE = "prediction_result_sub.xlsx"
PREDICTION_NOISE_FILE = "prediction_result_noise.xlsx"
PREDICTION_MIDCLASS_FILE = "prediction_result_mid.xlsx"

# KorPatELECTRA 예측 결과 파일 경로 설정
PREDICTION_SUBCLASS_FILE = "prediction_result_el_sub.xlsx"
PREDICTION_NOISE_FILE = "prediction_result_el_noise.xlsx"
PREDICTION_MIDCLASS_FILE = "prediction_result_el_mid.xlsx"
#----------------------------------------------------

ANSWER_FILE = "patent_data_finalN.csv"



# ===== 정답 데이터 불러오기 =====
df_true = pd.read_csv(ANSWER_FILE, encoding="cp949")
df_true = df_true[df_true['label'].notnull()].copy().reset_index(drop=True)

# 공통 텍스트 기준 열 생성 (title + korean_summary)
df_true["text"] = df_true["title"].fillna("") + " " + df_true["korean_summary"].fillna("")

# 텍스트 기반 매칭을 위한 키 생성
def normalize_text(text):
    return str(text).strip().replace(" ", "")

df_true["match_key"] = df_true["text"].apply(normalize_text)


# ===== 예측 결과 불러오기 함수 =====
def load_prediction(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df["text"] = df["title"].fillna("") + " " + df["korean_summary"].fillna("")
    df["match_key"] = df["text"].apply(normalize_text)
    return df


# ===== 정확도 비교 함수: Sub (정확히 일치) =====
def evaluate_sub(pred_df):
    merged = pd.merge(pred_df, df_true, on="match_key", suffixes=("_pred", "_true"))
    merged = merged.drop_duplicates(subset="match_key")
    merged = merged[merged["label_true"].str.strip() != ""]
    correct = (merged["label_pred"] == merged["label_true"]).sum()
    total = len(merged)
    acc = correct / total
    print("\n📘 [소분류] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 정확도 비교 함수: Noise ('N'인지 여부만 비교) =====
def evaluate_noise(pred_df):
    merged = pd.merge(pred_df, df_true, on="match_key", suffixes=("_pred", "_true"))
    merged = merged.drop_duplicates(subset="match_key")
    merged = merged[merged["label_true"].str.strip() != ""]

    def is_correct(pred, true):
        if pred == 'N':
            return true == 'N'
        else:
            return true != 'N'

    correct = sum(is_correct(p, t) for p, t in zip(merged["label_pred"], merged["label_true"]))
    total = len(merged)
    acc = correct / total
    print("\n📕 [노이즈] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 정확도 비교 함수: Mid (N이면 N 비교, 아니면 앞 두자리 비교) =====
def evaluate_mid(pred_df):
    merged = pd.merge(pred_df, df_true, on="match_key", suffixes=("_pred", "_true"))
    merged = merged.drop_duplicates(subset="match_key")
    merged = merged[merged["label_true"].str.strip() != ""]

    def is_correct(pred, true):
        if pred == 'N':
            return true == 'N'
        else:
            return true[:2] == pred

    correct = sum(is_correct(p, t) for p, t in zip(merged["label_pred"], merged["label_true"]))
    total = len(merged)
    acc = correct / total
    print("\n📙 [중분류] 정확도: %.4f (%d / %d)" % (acc, correct, total))


# ===== 실행 =====
evaluate_noise(load_prediction(PREDICTION_NOISE_FILE))
evaluate_sub(load_prediction(PREDICTION_SUBCLASS_FILE))
evaluate_mid(load_prediction(PREDICTION_MIDCLASS_FILE))
