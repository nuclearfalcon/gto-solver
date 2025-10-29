# Algorithm Comparison Report

**Generated:** 2025-10-29 00:02:59

## Game Configuration

- **Players:** 2
- **Stack sizes:** [500, 500]
- **Blinds:** [100, 50]
- **Betting abstraction:** fchpa
- **Effective stack (BB):** 5.0
- **Target iterations:** 100,000

## Results Summary

| Algorithm | Final Exploit | Time (s) | Avg Speed (it/s) | Peak Memory (MB) |
|-----------|---------------|----------|------------------|------------------|
| C++ CFR+ | 0.000032 | 2330.5 | 32.2 | 59.4 |
| External Sampling MCCFR | 5.689059 | 116.1 | 656.5 | 60.4 |

## Best Performers

- **Lowest Exploitability:** C++ CFR+ (0.000032)
- **Fastest (it/s):** External Sampling MCCFR (656.5 it/s)
- **Lowest Memory:** C++ CFR+ (59.4 MB)

## Recommendations

Based on these results:

1. For best convergence (lowest exploitability), use: **C++ CFR+**
2. For fastest iterations, use: **External Sampling MCCFR**
3. For lowest memory usage, use: **C++ CFR+**

Note: C++ implementations generally provide 5-10x speedup over Python versions.
MCCFR algorithms use less memory but may require more iterations to converge.
