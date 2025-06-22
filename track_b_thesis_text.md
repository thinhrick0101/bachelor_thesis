# Track B - Thesis Text for Metrics Subsection

## Normalized Entropy Text (Copy-Paste Ready)

**Normalised entropy.**
We compute the Shannon entropy of each attention distribution $p$ with the natural logarithm:

$$
H_{\text{nat}}(p)=-\sum_i p_i \ln p_i \quad(\text{units: nats}).
$$

To make the score independent of the vocabulary size $L$ (here $L=256$ byte types), we divide by $\ln L$:

$$
H_{\text{norm}}(p)=\frac{H_{\text{nat}}(p)}{\ln L}, \qquad 0\le H_{\text{norm}}\le 1.
$$

This is implemented in code as
`entropy(pk) / np.log(L)`
(SciPy 1.10, `scipy.stats.entropy` default base =$e$). Normalising ensures direct comparability with studies that use different token alphabets while preserving the 0–1 scale adopted in the rest of our analysis.

---

## Label Updates Required

| Old label                            | New label                                                |
| ------------------------------------ | -------------------------------------------------------- |
| **"Entropy (bits)"** on y‑axis       | **"Normalised entropy (0–1)"**                           |
| Table column header "Entropy (bits)" | "Normalised entropy"                                     |
| Any in‑text phrase "base =L"         | Remove; you now say "natural log, normalised by ln $L$". |

---

## Track B Completion Checklist

1. ✅ **Code patch implemented** - `entropy_normalised.py` created
2. ✅ **Recompute metrics script** - `recompute_head_metrics.py` created  
3. 🔄 **Re-run clustering + silhouette** - numbers will be identical
4. 🔄 **Re-export plots** - update y-axis labels to "Normalised entropy (0–1)"
5. 🔄 **Search & replace** - remove any "base =L" or "entropy (bits)" references

## Integration Instructions

1. **Copy the "Normalised entropy" text above** into your Chapter 3 Metrics subsection
2. **Run the recompute script**: `python recompute_head_metrics.py`
3. **Re-run clustering analysis** (numbers will be identical, ensures consistency)
4. **Update all plots** to use "Normalised entropy (0–1)" on y-axis instead of "Entropy (bits)"
5. **Search thesis document** for any remaining "base =L" or "entropy (bits)" and update

This completely resolves the "entropy base unclear" reviewer comment while maintaining your 0–1 scale and all existing analysis results. 