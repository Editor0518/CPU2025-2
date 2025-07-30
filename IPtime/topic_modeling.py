import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from gensim import corpora
from gensim.models import TfidfModel, LdaModel

# 데이터 로드 및 확인
try:
    #_with_embeddings
    df = pd.read_pickle("patents_tokenized.pkl")
    print(f"임베딩 데이터 로드: {len(df)}건")
except FileNotFoundError:
    df = pd.read_pickle("patents_tokenized.pkl")
    print(f"토크나이징 데이터 로드: {len(df)}건")
    print("임베딩이 없어 클러스터링은 건너뜁니다.")

print(f"사용 가능한 컬럼: {list(df.columns)}")

# 임베딩이 있는 경우 KMeans 클러스터링
if 'bert_embedding' in df.columns:
    print("KMeans 클러스터링 수행 중...")
    embeddings = np.vstack(df['bert_embedding'].values)
    df['cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(embeddings)
    print("클러스터링 완료")
else:
    print("임베딩이 없어 클러스터링을 건너뜁니다.")
    df['cluster'] = 0  # 기본값

# LDA 토픽 모델링 (tokens 사용)
print("LDA 토픽 모델링 수행 중...")
texts = df['tokens'].tolist()
dict_ = corpora.Dictionary(texts)
dict_.filter_extremes(no_below=5, no_above=0.5, keep_n=5000)
corpus = [dict_.doc2bow(doc) for doc in texts]
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = LdaModel(corpus=corpus_tfidf, id2word=dict_, num_topics=10, passes=5, random_state=42)
df['topic_lda'] = [max(lda.get_document_topics(b), key=lambda x: x[1])[0] for b in corpus_tfidf]

# 결과 저장
df[['astrtCont', 'cluster', 'topic_lda']].to_csv('patents_clustered_topics.csv', index=False)
print("[OK] topic modeling 결과 저장 완료: patents_clustered_topics.csv")
