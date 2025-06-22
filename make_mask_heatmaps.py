import torch, matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Mask factory (same logic as in the pseudocode) -------------------------
def build_mask(cluster_id, seq_len=256):   # use 256 for clearer plots
    i = torch.arange(seq_len).view(-1, 1)
    j = torch.arange(seq_len).view(1, -1)
    if cluster_id == 0:
        # Focused-local mask: half-width = 8 (full = 16)
        return (torch.abs(i - j) <= 8)
    if cluster_id == 1:
        return (i % 8) == (j % 8)
    if cluster_id == 2:
        anchors = torch.arange(0, seq_len, 64)
        local   = torch.abs(i - j) <= 4
        global_ = (j.unsqueeze(-1) == anchors.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        return local | global_
    if cluster_id == 3:
        return (torch.abs(i - j) <= 16)
    raise ValueError("unknown cluster")

# --- Plot settings ----------------------------------------------------------
cmap = ListedColormap(["black", "white"])      # 0 = black, 1 = white
titles = ["(a) Cluster 0 – Focused-Local",
          "(b) Cluster 1 – Strided",
          "(c) Cluster 2 – Global-Anchor",
          "(d) Cluster 3 – Wider-Local"]

plt.figure(figsize=(10, 8))
for k in range(4):
    plt.subplot(2, 2, k+1)
    mask = build_mask(k).int()                 # 0/1 tensor
    plt.imshow(mask, cmap=cmap, interpolation="nearest")
    plt.title(titles[k], fontsize=10)
    plt.xticks([]); plt.yticks([])

plt.tight_layout()
plt.savefig("attention_masks_heatmap.png", dpi=300)
print("✓ Saved figure as attention_masks_heatmap.png") 