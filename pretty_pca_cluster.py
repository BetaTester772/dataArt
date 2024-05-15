import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import numpy as np

# Load the sample data
data = pd.read_csv('solvedac.csv')

# Convert string representations of lists back to lists
data['tags'] = data['tags'].apply(eval)

# Binarize tags
mlb = MultiLabelBinarizer()
tags_encoded = mlb.fit_transform(data['tags'])

# Apply PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(tags_encoded)
data['pca-2d-one'] = pca_results[:, 0]
data['pca-2d-two'] = pca_results[:, 1]

# Apply KMeans clustering to PCA results
n_clusters = 4  # You can choose the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(pca_results)

# Generate a color palette with distinct colors
palette = sns.color_palette("husl", n_clusters)
colors = np.array([palette[cluster] for cluster in data['cluster']])

# Create a plot for data art
plt.figure(figsize=(30, 30))
scatter = plt.scatter(data['pca-2d-one'], data['pca-2d-two'], c=colors, s=5000, alpha=0.5,
                      edgecolors='none', linewidths=0.5)

# Remove ticks and labels for a cleaner look
plt.xticks([])
plt.yticks([])
plt.box(False)

# Show the plot
plt.show()
