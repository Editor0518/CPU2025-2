import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer

# 설정
vocab_path = "./pretrained/korpat_vocab.txt"
max_len = 256

def process_text(text, tokenizer, max_len):
    """텍스트를 토크나이징"""
    if pd.isna(text) or text.strip() == "":
        return [0]*max_len, [0]*max_len, []
    input_ids, segment_ids = tokenizer.encode(text, max_len=max_len)
    tokens = tokenizer.tokenize(text)
    return input_ids, segment_ids, tokens

def run_program():
    search_word = entry_search.get().strip()
    withdrawn_flag = var_withdrawn.get()

    if not search_word:
        messagebox.showwarning("입력 오류", "검색식을 입력하세요.")
        return

    if not os.path.exists(vocab_path):
        messagebox.showerror("오류", f"{vocab_path} 파일이 없습니다.")
        return

    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, "[INFO] KorPatBERT 토크나이저 초기화 중...\n")

    tokenizer = Tokenizer(vocab_path=vocab_path, cased=True)
    text_output.insert(tk.END, "[OK] 토크나이저 초기화 완료\n")

    text_output.insert(tk.END, f"🔍 검색어: {search_word} | 취하 포함: {withdrawn_flag}\n")
    text_output.insert(tk.END, "[INFO] 특허 검색 중...\n")

    try:
        df = fetch_patents(search_word=search_word, year_to_search="0", withdrawn=withdrawn_flag)
    except Exception as e:
        messagebox.showerror("에러", f"API 호출 실패: {e}")
        return

    if df.empty:
        text_output.insert(tk.END, "[WARN] 검색 결과가 없습니다.\n")
        return

    text_output.insert(tk.END, f"[OK] {len(df)}건 수집 완료\n")
    text_output.insert(tk.END, "[INFO] 토크나이징 진행 중...\n")

    results = df['astrtCont'].fillna('').apply(lambda x: process_text(x, tokenizer, max_len))
    df['input_ids'] = results.apply(lambda x: x[0])
    df['segment_ids'] = results.apply(lambda x: x[1])
    df['tokens'] = results.apply(lambda x: x[2])

    output_file = "patents_tokenized.pkl"
    df.to_pickle(output_file)
    text_output.insert(tk.END, f"[OK] 저장 완료: {output_file}\n")

    # 통계
    text_output.insert(tk.END, "\n=== 통계 ===\n")
    token_lengths = df['tokens'].apply(len)
    text_output.insert(tk.END, f"총 특허 건수: {len(df)}\n")
    text_output.insert(tk.END, f"평균 토큰 수: {token_lengths.mean():.1f}\n")
    text_output.insert(tk.END, f"최대 토큰 수: {token_lengths.max()}\n")
    text_output.insert(tk.END, f"최소 토큰 수: {token_lengths.min()}\n")

    # 샘플 출력
    first_patent = df.iloc[0]
    title = str(first_patent.get('inventionTitle', 'N/A'))[:50]
    content = str(first_patent.get('astrtCont', ''))
    tokens = first_patent['tokens']
    input_ids = first_patent['input_ids']

    text_output.insert(tk.END, "\n=== 샘플 ===\n")
    text_output.insert(tk.END, f"발명명: {title}...\n")
    text_output.insert(tk.END, f"원문 길이: {len(content)}\n")
    text_output.insert(tk.END, f"토큰 수: {len(tokens)}\n")
    text_output.insert(tk.END, f"토큰 예시: {tokens[:10]}\n")
    text_output.insert(tk.END, f"input_ids 예시: {input_ids[:10]}\n")

    text_output.insert(tk.END, "\n[완료] 토크나이징이 성공적으로 완료되었습니다.\n")

# GUI 구성
root = tk.Tk()
root.title("KIPRIS 특허 수집 + KorPatBERT 토크나이징")

# 검색식 입력
tk.Label(root, text="검색식:").grid(row=0, column=0, sticky="e")
entry_search = tk.Entry(root, width=40)
entry_search.grid(row=0, column=1, padx=5, pady=5)

# 옵션: 취하/소멸 포함 여부
var_withdrawn = tk.BooleanVar()
chk_withdrawn = tk.Checkbutton(root, text="취하/소멸/포기 포함", variable=var_withdrawn)
chk_withdrawn.grid(row=1, column=1, sticky="w", padx=5)

# 실행 버튼
btn_run = tk.Button(root, text="실행", command=run_program)
btn_run.grid(row=2, column=1, sticky="e", padx=5, pady=5)

# 출력창
text_output = tk.Text(root, width=80, height=30)
text_output.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
