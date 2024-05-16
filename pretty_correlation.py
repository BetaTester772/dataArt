import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import ast

df = pd.read_csv('solvedac.csv')

# 태그 데이터를 실제 리스트로 변환
df['tags'] = df['tags'].apply(ast.literal_eval)

# 이진 인코딩 수행
mlb = MultiLabelBinarizer()
tags_encoded = mlb.fit_transform(df['tags'])
tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

# 상관관계 계산
correlation_matrix = tags_df.corr()

# 상관관계 시각화
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=False, cmap='Pastel1', fmt=".2f", cbar=False)  # cbar=False로 컬러 바를 삭제

plt.show()
