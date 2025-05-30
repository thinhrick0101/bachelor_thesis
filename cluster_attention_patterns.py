import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def load_attention_patterns(file_path):
    """Load and parse attention pattern statistics from the analysis file."""
    patterns = {
        'layer': [],
        'head': [],
        'avg_attention': [],
        'entropy': [],
        'max_attention': [],
        'sparsity_90': []
    }
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError(f"File {file_path} is empty")
    
    print(f"Found {len(lines)} lines in file")
    
    current_head = None
    current_layer = None
    current_stats = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Try to match head number
        head_match = re.match(r'Head (\d+):', line)
        if head_match:
            # If we have collected stats for a previous head, save them
            if current_head is not None and current_stats:
                patterns['layer'].append(current_layer if current_layer is not None else len(patterns['layer']) // 16)
                patterns['head'].append(current_head)
                patterns['avg_attention'].append(current_stats.get('avg', 0.0))
                patterns['entropy'].append(current_stats.get('entropy', 0.0))
                patterns['max_attention'].append(current_stats.get('max', 0.0))
                patterns['sparsity_90'].append(current_stats.get('sparsity', 0.0))
            
            current_head = int(head_match.group(1))
            current_stats = {}
            continue
        
        # Try to match statistics
        avg_match = re.match(r'Average attention: ([\d.]+)', line)
        if avg_match:
            current_stats['avg'] = float(avg_match.group(1))
            continue
            
        entropy_match = re.match(r'Entropy: ([\d.-]+)', line)
        if entropy_match:
            current_stats['entropy'] = float(entropy_match.group(1))
            continue
            
        sparsity_match = re.match(r'Sparsity \(90% mass\): ([\d.]+)', line)
        if sparsity_match:
            current_stats['sparsity'] = float(sparsity_match.group(1))
            continue
            
        # Try to match layer number (if present)
        layer_match = re.match(r'Layer (\d+)', line)
        if layer_match:
            current_layer = int(layer_match.group(1))
            continue
    
    # Don't forget to add the last head
    if current_head is not None and current_stats:
        patterns['layer'].append(current_layer if current_layer is not None else len(patterns['layer']) // 16)
        patterns['head'].append(current_head)
        patterns['avg_attention'].append(current_stats.get('avg', 0.0))
        patterns['entropy'].append(current_stats.get('entropy', 0.0))
        patterns['max_attention'].append(current_stats.get('max', 0.0))
        patterns['sparsity_90'].append(current_stats.get('sparsity', 0.0))
    
    df = pd.DataFrame(patterns)
    
    if df.empty:
        raise ValueError("No data was parsed from the file. Check if the file format matches the expected pattern.")
    
    print(f"Successfully loaded {len(df)} attention heads across {df['layer'].nunique()} layers")
    
    # Sort by layer and head
    df = df.sort_values(['layer', 'head']).reset_index(drop=True)
    
    return df

def cluster_attention_heads(df, n_clusters=4):
    """Cluster attention heads based on their features."""
    # Prepare features for clustering
    features = ['avg_attention', 'entropy', 'max_attention', 'sparsity_90']
    X = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    return df, kmeans, X_scaled

def plot_cluster_distributions(df):
    """Plot the distribution of clusters across layers."""
    plt.figure(figsize=(12, 6))
    
    # Create a pivot table of cluster counts per layer
    cluster_dist = pd.crosstab(df['layer'], df['cluster'])
    
    # Plot stacked bar chart
    cluster_dist.plot(kind='bar', stacked=True)
    plt.title('Distribution of Attention Head Clusters Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Number of Heads')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('attention_clusters_distribution.png')
    plt.close()

def plot_cluster_characteristics(df):
    """Plot the characteristics of each cluster."""
    features = ['avg_attention', 'entropy', 'max_attention', 'sparsity_90']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        sns.boxplot(data=df, x='cluster', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature} by Cluster')
    
    plt.tight_layout()
    plt.savefig('attention_cluster_characteristics.png')
    plt.close()

def plot_2d_projections(X_scaled, clusters, method='pca'):
    """Plot 2D projections of the clusters using PCA or t-SNE."""
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Projection of Attention Head Clusters'
    else:
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Projection of Attention Head Clusters'
    
    X_2d = reducer.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.savefig(f'attention_clusters_{method}.png')
    plt.close()

def analyze_clusters(df):
    """Generate a textual analysis of the clusters."""
    analysis = []
    
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        
        # Calculate mean characteristics
        means = cluster_data[['avg_attention', 'entropy', 'max_attention', 'sparsity_90']].mean()
        
        # Determine cluster type based on characteristics
        characteristics = []
        if means['entropy'] < df['entropy'].mean():
            characteristics.append("focused")
        else:
            characteristics.append("diffuse")
            
        if means['max_attention'] > df['max_attention'].mean():
            characteristics.append("peaky")
            
        if means['sparsity_90'] > df['sparsity_90'].mean():
            characteristics.append("sparse")
        else:
            characteristics.append("uniform")
        
        # Analyze layer distribution
        layer_dist = cluster_data['layer'].value_counts().sort_index()
        if layer_dist.index[0] < 4:
            layer_location = "early"
        elif layer_dist.index[-1] > 8:
            layer_location = "late"
        else:
            layer_location = "middle"
        
        analysis.append(f"""
Cluster {cluster}:
- Characteristics: {', '.join(characteristics)}
- Average attention: {means['avg_attention']:.3f}
- Entropy: {means['entropy']:.3f}
- Max attention: {means['max_attention']:.3f}
- Sparsity (90%): {means['sparsity_90']:.3f}
- Primarily found in {layer_location} layers
- Number of heads: {len(cluster_data)}
""")
    
    return '\n'.join(analysis)

def main():
    try:
        # Load attention patterns
        print("Loading attention patterns...")
        file_path = 'attention_patterns_analysis_d.txt'
        
        # First try in current directory
        if not Path(file_path).exists():
            # Try in bachelor_thesis directory
            alt_path = Path('bachelor_thesis') / file_path
            if alt_path.exists():
                file_path = str(alt_path)
            else:
                print(f"Searching for file in current directory: {Path.cwd()}")
                print("Available files:")
                for f in Path.cwd().glob('*.txt'):
                    print(f"  - {f}")
        
        df = load_attention_patterns(file_path)
        
        # Cluster the attention heads
        print("\nClustering attention heads...")
        df, kmeans, X_scaled = cluster_attention_heads(df)
        
        # Create output directory
        output_dir = Path('attention_analysis')
        output_dir.mkdir(exist_ok=True)
        print(f"\nCreated output directory: {output_dir}")
        
        # Generate and save visualizations
        print("\nGenerating visualizations...")
        plot_cluster_distributions(df)
        plot_cluster_characteristics(df)
        plot_2d_projections(X_scaled, df['cluster'], method='pca')
        plot_2d_projections(X_scaled, df['cluster'], method='tsne')
        
        # Generate cluster analysis
        print("\nGenerating cluster analysis...")
        analysis = analyze_clusters(df)
        
        # Save analysis to file
        analysis_file = output_dir / 'cluster_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write(analysis)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 