#!/usr/bin/env python3
"""
Create head_labels.csv from head_metrics.csv by assigning cluster labels
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

print("🎯 Creating head_labels.csv for Results & Visuals")
print("=" * 50)

# Load the head metrics
df = pd.read_csv("head_metrics.csv")
print(f"📊 Loaded {len(df)} attention heads")

# Prepare features for clustering
X = df[["entropy", "sparsity", "distance"]].values
X_scaled = StandardScaler().fit_transform(X)

# Perform clustering with k=4
clusterer = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = clusterer.fit_predict(X_scaled)

# Create head_labels dataframe
head_labels = pd.DataFrame({
    'head_id': range(len(df)),  # Sequential head ID
    'layer': df['layer'],
    'head': df['head'],
    'cluster': cluster_labels
})

# Save the file
head_labels.to_csv("head_labels.csv", index=False)
print(f"✅ Saved head_labels.csv with {len(head_labels)} entries")

# Show preview
print(f"\n📋 Preview of head_labels.csv:")
print(head_labels.head(10))

# Show cluster distribution
print(f"\n📊 Cluster Distribution:")
cluster_counts = head_labels["cluster"].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"   Cluster {cluster}: {count} heads")

print(f"\n🎯 Ready for Results & Visuals generation!") 