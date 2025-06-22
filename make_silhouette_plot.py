import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# 1) Load per-head metrics  (replace filename as needed)
df = pd.read_csv("head_metrics.csv")          # columns: entropy, sparsity, distance

# 2) Feature matrix and z-score normalisation
X = df[["entropy", "sparsity", "distance"]].values
X = StandardScaler().fit_transform(X)

# 3) Compute silhouette for k = 2 … 8
k_range, sil_scores = range(2, 9), []
for k in k_range:
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

# 4) Plot
plt.figure(figsize=(6,4))
plt.plot(k_range, sil_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average silhouette score")
plt.title("Silhouette profile vs. k")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_k_plot.png", dpi=300)
print("✓ Saved plot as silhouette_k_plot.png") 