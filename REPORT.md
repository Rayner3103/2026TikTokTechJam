# Rayner Final Model Report

Generated: 2026-09-01T01:08:01.021131+00:00

The final model combines SDPA, GQA, fused SiLU GLU, and a training-ready adaptive router. The router is disabled for this inference sweep. Each case compares independently initialized models on the same GPU input. Cross-model numerical accuracy is not claimed because GQA and GLU change the parameter topology; output shape and finite-value checks are used.

## Results

| # | Batch | D Model | Heads | Seq Len | Layers | FFN | Baseline ms | Final ms | Speedup | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | 4.5416 | 4.4846 | 1.013x | PASS, contract PASS |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | 2.0470 | 2.5584 | 0.800x | PASS, contract PASS |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | 2.1688 | 2.4069 | 0.901x | PASS, contract PASS |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | 1.8930 | 2.6490 | 0.715x | PASS, contract PASS |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | 8.6349 | 7.8566 | 1.099x | PASS, contract PASS |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | N/A | N/A | N/A | FAILED |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | 2.9215 | 2.4090 | 1.213x | PASS, contract PASS |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | 37.6161 | 47.0328 | 0.800x | PASS, contract PASS |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | 2.5852 | 3.6490 | 0.708x | PASS, contract PASS |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | 3.4084 | 3.8508 | 0.885x | PASS, contract PASS |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | 9.2636 | 4.5149 | 2.052x | PASS, contract PASS |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | 1.8687 | 2.4084 | 0.776x | PASS, contract PASS |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | 126.1399 | 53.6540 | 2.351x | PASS, contract PASS |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | N/A | N/A | N/A | FAILED |

## Interpretation

- Successful runs: 12/14.
- Positive speedup means the final model had lower median latency than the baseline.
- Cases that fail with CUDA OOM are recorded as resource limits, not model correctness failures.
- Per-case logs are saved as `run_config_<n>.log` beside this report.

## Reproduction

```bash
cd Rayner
python run_final_sweep.py
```
