# topic_wordcloud.py

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import os
from korpat_tokenizer import Tokenizer

# 출력 디렉토리
output_dir = "output_wordclouds"
os.makedirs(output_dir, exist_ok=True)

# CSV 파일 로드
csv_file = "patents_clustered_topics.csv"
df = pd.read_csv(csv_file)

# 마스크 이미지 로드 (에러 방지를 위해 try-except 추가)
try:
    mask_url = "https://media.cheggcdn.com/media/216/21621ee5-e80f-47f3-9145-513f2229b390/phploeBuh.png"
    response = requests.get(mask_url)
    mask_img = np.array(Image.open(BytesIO(response.content)).convert("L"))
    print("[OK] 마스크 이미지 로드 완료")
except Exception as e:
    print(f"[WARN] 마스크 이미지 로드 실패: {e}")
    mask_img = None


# stopwords.txt 파일 읽기
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

stopwords = load_stopwords("stopwords.txt")
print(f"[OK] {len(stopwords)}개의 stopwords 로드 완료")


# KorPatBERT 토크나이저 초기화
tokenizer = Tokenizer(vocab_path="./pretrained/korpat_vocab.txt", cased=True)
print("[OK] KorPatBERT 토크나이저 초기화 완료")

# WordCloud 생성 함수
def generate_wordcloud(text_list, title, filename):
    words = []
    for txt in text_list:
        if isinstance(txt, str) and txt.strip():
            # 텍스트 전처리 및 단어 분할
            import re
            
            # 특수문자 제거 및 정리
            clean_text = re.sub(r'[^\w\s가-힣]', ' ', txt)
            clean_text = re.sub(r'\s+', ' ', clean_text)  # 연속 공백 제거
            
            # 공백으로 분할
            split_words = clean_text.split()
            
            # 토크나이저도 시도 (실패하면 공백 분할 결과 사용)
            try:
                morphs = tokenizer._get_morphs(txt)
                # 형태소 분석 결과가 의미있는 경우에만 사용
                if len(morphs) > len(split_words) * 0.5:  # 분할된 단어 수의 50% 이상이면 사용
                    words.extend(morphs)
                else:
                    words.extend(split_words)
            except:
                words.extend(split_words)
    
    # 필터링 전 단어 수 출력
    print(f"\n=== {title} 토크나이징 결과 ===")
    print(f"원본 단어 수: {len(words)}")
    
    # 필터링 적용 (더 엄격하게)
    filtered_words = [
        w for w in words 
        if (w not in stopwords and 
            len(w) > 1 and 
            len(w) < 20 and  # 너무 긴 단어 제외
            not w.isdigit() and  # 숫자만으로 구성된 단어 제외
            not re.match(r'^[a-zA-Z]+$', w))  # 영문만으로 구성된 단어 제외
    ]
    print(f"필터링 후 단어 수: {len(filtered_words)}")
    
    if len(filtered_words) > 0:
        word_freq = Counter(filtered_words)
        print(f"상위 20개 단어: {dict(word_freq.most_common(20))}")
        
        # WordCloud 생성 시 오류 방지
        try:
            wc = WordCloud(
                font_path="c:/Windows/Fonts/NanumGothic.ttf",
                background_color="white",
                max_words=50,
                width=800,
                height=600,
                mask=mask_img,  # ✅ mask_img를 사용
                random_state=39,
                min_font_size=10,
                max_font_size=100,
                relative_scaling=0.5
            )

            wc.generate_from_frequencies(word_freq)

            plt.figure(figsize=(10, 8))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] 워드클라우드 저장 완료: {filename}")
            
        except Exception as e:
            print(f"[ERROR] 워드클라우드 생성 실패 ({title}): {e}")
            # 대신 단어 빈도 텍스트 파일 저장
            with open(os.path.join(output_dir, f"{filename.replace('.png', '.txt')}"), 'w', encoding='utf-8') as f:
                f.write(f"{title} - 단어 빈도\n")
                f.write("=" * 50 + "\n")
                for word, freq in word_freq.most_common(50):
                    f.write(f"{word}: {freq}\n")
            print(f"[OK] 단어 빈도 텍스트 파일 저장: {filename.replace('.png', '.txt')}")
    else:
        print(f"[WARN] {title}에는 유효한 단어가 없어 워드클라우드를 생성하지 않습니다.")
        # 빈 결과도 기록
        with open(os.path.join(output_dir, f"{filename.replace('.png', '_empty.txt')}"), 'w', encoding='utf-8') as f:
            f.write(f"{title} - 단어가 없음\n")
            f.write("필터링된 모든 단어가 stopwords이거나 조건에 맞지 않습니다.\n")

# 클러스터별 생성
for cluster_id in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster_id]
    generate_wordcloud(subset['astrtCont'], f"Cluster {cluster_id}", f"wordcloud_cluster_{cluster_id}.png")

# 토픽별 생성
for topic_id in sorted(df['topic_lda'].unique()):
    subset = df[df['topic_lda'] == topic_id]
    generate_wordcloud(subset['astrtCont'], f"Topic {topic_id}", f"wordcloud_topic_{topic_id}.png")

print("\n✔ 모든 워드클라우드 생성 완료")
