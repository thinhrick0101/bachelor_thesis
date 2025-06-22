# Model Performance Comparison

| model   | loss          | ppl           | speed          | mem                |
|:--------|:--------------|:--------------|:---------------|:-------------------|
| dense   | 0.872 ± 0.000 | 2.393 ± 0.000 | 21.920 ± 0.000 | 195303.753 ± 0.000 |
| sparse  | 1.005 ± 0.000 | 2.731 ± 0.000 | 85.304 ± 0.000 | 163252.193 ± 0.000 |

**Table**: Performance comparison between dense and sparse transformer models. Lower is better for Loss, PPL, and Peak Memory; higher is better for Tokens/sec.
