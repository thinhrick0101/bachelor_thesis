
# WORD INTEGRATION GUIDE
## Results & Visuals Package

### FILES GENERATED:
1. table_cluster_counts.tsv - Basic cluster counts
2. table_cluster_counts_named.tsv - Cluster counts with names
3. table_cluster_summary.tsv - Summary statistics by cluster
4. fig3_2_layer_cluster_heatmap.png - Layer × Cluster heatmap
5. fig3_3_attention_patterns_combined.png - 2×2 attention patterns
6. cluster0_example.png through cluster3_example.png - Individual patterns

### WORD INTEGRATION STEPS:

#### Table 1: Cluster Distribution
1. Open table_cluster_counts_named.tsv in text editor
2. Ctrl+A → Ctrl+C
3. In Word → Paste → "Keep Text Only"
4. Select pasted text → Insert ▸ Table ▸ Convert Text to Table (delimiter = Tabs)
5. Caption: "Table 4.1. Distribution of attention heads across four discovered clusters."

#### Figure 3.2: Layer × Cluster Heatmap
1. Insert ▸ Pictures ▸ This Device → select fig3_2_layer_cluster_heatmap.png
2. Caption: "Figure 3.2. Distribution of attention head clusters across transformer layers. 
   Numbers indicate count of heads per layer-cluster combination."

#### Figure 3.3: Attention Patterns
1. Insert ▸ Pictures ▸ This Device → select fig3_3_attention_patterns_combined.png
2. Caption: "Figure 3.3. Representative attention patterns for each discovered cluster. 
   (a) Focused-Local: short-range dependencies, (b) Strided: regular intervals, 
   (c) Global-Anchor: local + global positions, (d) Wider-Local: medium-range dependencies."

#### Table 2: Summary Statistics (Optional)
1. Open table_cluster_summary.tsv in text editor
2. Follow same steps as Table 1
3. Caption: "Table 4.2. Mean ± standard deviation of attention metrics by cluster."

### TEXT REFERENCES:
- "As shown in Table 4.1, the four clusters contain 28, 24, 20, and 24 heads respectively..."
- "Figure 3.2 reveals that Cluster 0 (Focused-Local) dominates early layers..."
- "The qualitative differences are visualized in Figure 3.3, which shows..."

### CHECKLIST:
□ Table 4.1 inserted and referenced
□ Figure 3.2 inserted and referenced  
□ Figure 3.3 inserted and referenced
□ All captions added
□ Text references updated
