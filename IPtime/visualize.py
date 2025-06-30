# visualize.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from gensim import corpora
from gensim.models import LdaModel

# 1) 결과 불러오기
df = pd.read_csv('patents_clustered_topics.csv')

# 2) 임베딩 재로딩 (필요 시)
# 만약 embeddings를 별도 .npy 로 저장해 두셨다면:
# embeddings = np.load('embeddings.npy')

# 대신, silhouette 계산과 PCA는 topic_modeling.py 에서 미리 저장해두는 것이 좋습니다.
# 아래는 가정 예시: embeddings.npy 로 저장했다고 할 때
embeddings = np.load('embeddings.npy')

# 3) Silhouette score
score = silhouette_score(embeddings, df['cluster'])
print(f"Silhouette score: {score:.4f}")

# 4) PCA 2D 시각화
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c=df['cluster'], cmap='tab10', s=30)
plt.title("PCA 2D Clustering")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(label="cluster")
plt.show()

# 5) LDA 토픽별 키워드 시각화
# (topic_modeling.py 에서 dict_, lda 객체를 pickle 로 저장해 두고 읽어오는 식으로)
import pickle
with open('lda_model.pkl','rb') as f:
    lda_model, dictionary = pickle.load(f)

for t in range(lda_model.num_topics):
    print(f"\nTopic {t}:")
    terms = lda_model.show_topic(t, topn=10)
    words, weights = zip(*terms)
    plt.figure(figsize=(4,2))
    plt.barh(words, weights)
    plt.gca().invert_yaxis()
    plt.title(f"Topic {t}")
    plt.show()
