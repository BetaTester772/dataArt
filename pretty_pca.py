import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

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

# Normalize difficulty levels for color mapping
norm = plt.Normalize(data['level'].min(), data['level'].max())
colors = plt.cm.Accent(norm(data['level']))

# Create a plot for data art
plt.figure(figsize=(30, 30))
scatter = plt.scatter(data['pca-2d-one'], data['pca-2d-two'], c=colors, cmap='Paired', s=5000, alpha=0.5,
                      edgecolors='none', linewidths=0.5)

# Remove ticks and labels for a cleaner look
plt.xticks([])
plt.yticks([])
plt.box(False)

# Show the plot
plt.show()
