import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from adjustText import adjust_text

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
data = pd.read_csv('solvedac.csv')

# 태그의 문자열을 리스트로 변환
data['tags'] = data['tags'].apply(eval)

# 태그를 이진 인코딩
mlb = MultiLabelBinarizer()
tags_encoded = mlb.fit_transform(data['tags'])

# PCA 적용
pca = PCA(n_components=2)
pca_results = pca.fit_transform(tags_encoded)
data['pca-2d-one'] = pca_results[:, 0]
data['pca-2d-two'] = pca_results[:, 1]

# 난이도에 따른 색상 지정
norm = plt.Normalize(1, 30)
colors = plt.cm.viridis(norm(data['level']))

# 점의 크기를 크게 설정
point_size = 100

# 제목이 있는 그래프
plt.figure(figsize=(15, 10))
scatter = plt.scatter(data['pca-2d-one'], data['pca-2d-two'], c=colors, cmap='viridis', s=point_size, alpha=0.6)
texts = []
for i, title in enumerate(data['title']):
    texts.append(plt.text(data['pca-2d-one'][i], data['pca-2d-two'][i], title, ha='center', va='center', fontsize=9))

# adjust_text로 텍스트 겹침 해결
adjust_text(texts,
            # expand_text=(1.2, 1.5),
            expand_points=(1.2, 1.5),
            force_text=(0.5, 0.5),
            force_points=(0.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            iterations=50)

plt.colorbar(scatter, label='Difficulty Level')
plt.title('Clustering of Problems by Tags with Difficulty Level, with Titles')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.grid(True)
plt.show()
