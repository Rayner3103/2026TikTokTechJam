# Exploration Benchmark Results

Generated from `Rayner/exploration/run_all.py` on 2026-09-01. The runner tested Directions 1-7 against the shared baseline on small, medium, and large configurations.

## Standard Results

| Direction | Small | Medium | Large | Accuracy |
|---|---:|---:|---:|---|
| 1: SDPA | 0.97x | 0.99x | 1.04x | 3/3 PASS |
| 2: QKV fusion | 0.77x | 1.00x | 1.02x | 3/3 PASS |
| 3: max-autotune | 1.01x | 1.01x | 1.03x | 3/3 PASS |
| 4: fused GLU | 1.01x | 1.01x | 0.97x | 3/3 PASS |
| 5: GQA + RoPE | 0.99x | 1.02x | 1.00x | 3/3 PASS |
| 6: sliding window | 1.04x | 1.03x | 0.99x | 3/3 PASS |
| 7: adaptive routing | 1.12x | 1.03x | 0.99x | 3/3 PASS |

**Standard accuracy: 21/21 checks passed.** Speedups are median-latency ratios relative to the baseline; values above 1.0x are faster.

## Additional Validation

The runner also executed five baseline validation scenarios after the direction sweep:

| Scenario | Result | Speedup |
|---|---|---:|
| `causal_padded` | PASS | 1.011x |
| `float16_causal` | PASS | 1.015x |
| `bfloat16_padded` | PASS | 1.032x |
| `minimal_sequence` | PASS | 1.002x |
| `compiled_user` | PASS | 7.483x |

Per-direction validation attempts are retained in `results.json`. Directions 2-7 use different parameter topologies, so copying baseline weights is not a valid elementwise accuracy comparison for those implementations. The standard benchmark initializes each architecture correctly and is the reliable comparison above.

## Reproduction

```bash
cd Rayner/exploration
python run_all.py
```

Raw terminal output is stored in `exploration_benchmark_latest.log`, and structured results are stored in `Rayner/exploration/results/results.json`.
