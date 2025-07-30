# main.py
# 1) 특허 데이터 불러오기, 2) KorPat 토크나이즈 → input_ids 생성
# (이전 예제에서 구현한 fetch_patents, encode_patents 가 그대로 필요합니다)
# topicmodeling.py

#3.8.20 usethis 를 환경으로 쓰기!!!!! pip

import numpy as np
# numpy 1.20+ 에서 제거된 np.object 별칭을 복원
setattr(np, 'object', object)

from patent_fetcher import fetch_patents
from korpat_tokenizer import Tokenizer
import pandas as pd

# clustering/topic modeling 에 필요한 라이브러리
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import gensim
from gensim import corpora
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from gensim.models import TfidfModel
from PIL import Image

# 이하 기존 코드…



setattr(np, 'object', object) # numpy 1.23 이상에서 object 타입이 deprecated 되어 오류 발생 방지

print("Pillow version:", Image.__version__)
plt.plot([1,2,3],[4,5,6])
plt.show()



def encode_patents(df, text_field, tokenizer, max_len=256):
    df['input_ids'], df['segment_ids'] = zip(*df[text_field]
        .fillna('')
        .apply(lambda txt: tokenizer.encode(txt, max_len=max_len))
    )
    return df


# 1) 특허 데이터 로드 & 토크나이즈
df = fetch_patents(
    search_word="(생성형 ai)*(게임 스토리)",
    year_to_search="0",
)

tokenizer = Tokenizer(vocab_path='./pretrained/korpat_vocab.txt', cased=True)

# 수정된 호출:
df = encode_patents(
    df,
    text_field='astrtCont',
    tokenizer=tokenizer,
    max_len=256
)

# 2) KorPatBERT 모델로부터 [CLS] 토큰 임베딩 추출 함수
def build_embed_model(config_dir, checkpoint_path, max_seq_len=256):
    """ BERT 레이어만 뽑아서 input_ids → cls_embedding 반환하는 Keras 모델 """
    # 설정 불러오기
    bert_params = bert.params_from_pretrained_ckpt(config_dir)
    bert_layer  = bert.BertModelLayer.from_params(bert_params, name="bert")
    # 입력 & BERT 연동
    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    seq_out   = bert_layer(input_ids)
    cls_token = keras.layers.Lambda(lambda x: x[:,0,:])(seq_out)  # [CLS]
    model     = keras.Model(inputs=input_ids, outputs=cls_token)
    # 가중치 로드
    bert.load_stock_weights(bert_layer, checkpoint_path)
    return model

# 모델 생성
embed_model = build_embed_model(
    config_dir="./pretrained/",
    checkpoint_path="./pretrained/model.ckpt-381250",
    max_seq_len=256
)

# 3) 임베딩 계산 (batch 단위로)
embeddings = embed_model.predict(
    np.vstack(df['input_ids'].values), 
    batch_size=8, 
    verbose=1
)  # shape = (N_documents, hidden_size)

# 4) K-Means 클러스터링
NUM_CLUSTERS = 5
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# (옵션) 2차원 시각화를 위해 PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
df['x'], df['y'] = coords[:,0], coords[:,1]

# 5) 형태소 분석 + Gensim LDA 토픽 모델링
#    MeCab으로 간단 토큰화 → Gensim Dictionary + Corpus → LDA
tokenizer = Tokenizer(vocab_path="./pretrained/korpat_vocab.txt", cased=True)
texts = [
    tokenizer._mecab.morphs(text if isinstance(text, str) else "") 
    for text in tqdm(df['astrtCont'], desc="MeCab morphs")
]

# 불용어 제거(간단 예)
stopwords = set(['및', '등', '있', '수', '것', '본', '발명', '서비스'])
texts = [[tok for tok in doc if tok not in stopwords] for doc in texts]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=5000)

corpus     = [dictionary.doc2bow(doc) for doc in texts]

NUM_TOPICS = 10

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=5,
    random_state=42
)

# 각 문서의 dominant topic 추출
doc_topics = []
for bow in corpus:
    topic_probs = lda.get_document_topics(bow)
    # 확률 가장 높은 토픽 번호
    top_topic = max(topic_probs, key=lambda x: x[1])[0]
    doc_topics.append(top_topic)
df['topic_lda'] = doc_topics

# 6) 결과 출력
print(df[['astrtCont','cluster','topic_lda']].head())

# (선택) 클러스터별 대표  문장, LDA 토픽별 키워드 출력
for c in range(NUM_CLUSTERS):
    print(f"\n--- Cluster {c} 대표 예시 ---")
    print(df[df['cluster']==c]['astrtCont'].iloc[0])

print("\n=== LDA 토픽 키워드 ===")
for t in range(NUM_TOPICS):
    print(f"토픽 {t}:", lda.print_topic(t))

# 7) 저장
df.to_csv('./patent-data/patents_clustered.csv', index=False)

score = silhouette_score(embeddings, df['cluster'])
print("Silhouette score:", score)

plt.figure(figsize=(6,6))
plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='tab10', s=30)
plt.title("PCA 2D Clustering")
plt.show()