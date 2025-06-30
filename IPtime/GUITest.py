import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------------------------------
# 1. Titanic 데이터 불러오기 (온라인 CSV)
# -------------------------------------------------------
# Github에 공개된 Titanic 데이터셋 CSV (승객 정보 포함)
CSV_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# pandas로 CSV를 읽어옵니다.
try:
    titanic_df = pd.read_csv(CSV_URL)
except Exception as e:
    tk.messagebox.showerror("데이터 로드 오류", f"Titanic 데이터를 불러오는 중 오류 발생:\n{e}")
    exit(1)

# -------------------------------------------------------
# 2. 시각화용 데이터 가공 (Pclass별 생존자 수 집계)
# -------------------------------------------------------
# 'Survived' 칼럼: 0 = 사망, 1 = 생존
# 'Pclass' 칼럼: 1, 2, 3 (선실 등급)
grouped = titanic_df.groupby("Pclass")["Survived"].sum().reset_index()
# 예: Pclass 1: 136, Pclass 2: 87, Pclass 3: 119 (생존자 수)

# -------------------------------------------------------
# 3. Tkinter 윈도우 설정
# -------------------------------------------------------
root = tk.Tk()
root.title("Tkinter + Matplotlib: Titanic Pclass별 생존자 수")
root.geometry("600x500")

# 상단 라벨
label = ttk.Label(root, text="Titanic 데이터: 선실 등급(Pclass)별 생존자 수", font=("맑은 고딕", 14))
label.pack(pady=10)

# -------------------------------------------------------
# 4. Matplotlib Figure 생성 및 막대그래프 그리기
# -------------------------------------------------------
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
classes = grouped["Pclass"].astype(str)  # ['1', '2', '3']
survivors = grouped["Survived"]

ax.bar(classes, survivors, color=["#4E79A7", "#F28E2B", "#E15759"])
ax.set_xlabel("선실 등급 (Pclass)")
ax.set_ylabel("생존자 수 (Survived)")
ax.set_title("Pclass별 생존자 수")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# 막대 위에 숫자 표시
for i, v in enumerate(survivors):
    ax.text(i, v + 3, str(int(v)), ha="center", fontweight="bold")

plt.tight_layout()

# -------------------------------------------------------
# 5. Matplotlib Figure를 Tkinter에 임베드
# -------------------------------------------------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# -------------------------------------------------------
# 6. 종료 버튼 (선택 사항)
# -------------------------------------------------------
def on_exit():
    root.destroy()

exit_btn = ttk.Button(root, text="닫기", command=on_exit)
exit_btn.pack(pady=5)

# -------------------------------------------------------
# 7. Tkinter 메인루프 시작
# -------------------------------------------------------
root.mainloop()
