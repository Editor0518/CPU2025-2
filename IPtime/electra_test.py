from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer 
model_name = "KIPI-ai/KorPatElectra"

# Access token (replace with your actual token from Hugging Face)
ACCESS_TOKEN_PATH = "access_token.txt"

# ===== ACCESS TOKEN 로딩 함수 =====
def load_token_from_file(path=ACCESS_TOKEN_PATH):
    with open(path, "r") as f:
        return f.read().strip()

access_token = load_token_from_file()

model = BertModel.from_pretrained(model_name, use_auth_token=access_token)
tokenizer = BertTokenizer.from_pretrained(model_name, use_auth_token=access_token)

# Sample sentence
sentence_org = "본 고안은 주로 일회용 합성세제액을 집어넣어 밀봉하는 세제액포의 내부를 원호상으로 열중착하되 세제액이 배출되는 절단부 쪽으로 내벽을 협소하게 형성하여서 내부에 들어있는 세제액을 잘짜질 수 있도록 하는 합성세제 액포에 관한 것이다."

# Tokenization
inputs = tokenizer(sentence_org, return_tensors="pt")

# Model input
outputs = model(**inputs)

# Extract the last hidden states
last_hidden_states = outputs.last_hidden_state
cls_vector = last_hidden_states[:, 0, :]  # (batch_size, hidden_size)

print(f"1. Length of vocab : {tokenizer.vocab_size}")
print(f"2. Input example : {sentence_org}")
print(f"3. Tokenized example : {inputs}")
print(f"4. vector shape : {cls_vector.shape}")
