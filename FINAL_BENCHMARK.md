# Final Model Benchmark

## Setup

The final composite model in [final_model.py](final_model.py) combines:

- Phase 1: SDPA attention.
- Phase 2: grouped-query attention and fused SiLU GLU FFN.
- Phase 3: adaptive routing, implemented but disabled for this sweep because the router is training-ready and untrained routing would make the comparison unstable.

Each case was run sequentially with [final_model_benchmark.py](final_model_benchmark.py) against the reference baseline using:

- Device: NVIDIA GeForce RTX 3080, CUDA.
- Dtype: FP32.
- Causal attention: enabled.
- Warmup: 3 iterations.
- Timing: 10 repetitions, median latency.
- Same generated input for baseline and final model.
- `kv_heads=4` for the 4-head cases; `kv_heads=heads` for 1- and 2-head cases.
- Accuracy was not compared because GQA and fused GLU change the parameter topology. Both models passed output-shape and finite-output checks for every successful case.

## Results

| # | Batch | QKV Dim | Heads | Seq Len | Layers | FFN Dim | Baseline ms | Final ms | Speedup | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 128 | 4 | 128 | 4 | 128 | 4.0929 | 4.0376 | **1.014x** | PASS |
| 2 | 1 | 128 | 4 | 128 | 4 | 128 | 1.8642 | 2.3436 | 0.795x | PASS |
| 3 | 4 | 128 | 4 | 128 | 4 | 128 | 1.8760 | 2.3740 | 0.790x | PASS |
| 4 | 16 | 128 | 4 | 128 | 4 | 128 | 2.0470 | 2.6139 | 0.783x | PASS |
| 5 | 128 | 128 | 4 | 128 | 4 | 128 | 8.5140 | 7.5832 | **1.123x** | PASS |
| 6 | 10000 | 128 | 4 | 128 | 4 | 128 | N/A | N/A | N/A | CUDA OOM |
| 7 | 64 | 32 | 4 | 128 | 4 | 32 | 2.5851 | 2.3859 | **1.083x** | PASS |
| 8 | 64 | 1024 | 4 | 128 | 4 | 1024 | 36.0187 | 45.2623 | 0.796x | PASS |
| 9 | 64 | 128 | 1 | 128 | 4 | 128 | 2.4632 | 3.8487 | 0.640x | PASS |
| 10 | 64 | 128 | 2 | 128 | 4 | 128 | 3.5842 | 3.6628 | 0.979x | PASS |
| 11 | 64 | 128 | 16 | 128 | 4 | 128 | 8.8684 | 4.4431 | **1.996x** | PASS |
| 12 | 64 | 128 | 4 | 32 | 4 | 128 | 1.8933 | 2.3861 | 0.793x | PASS |
| 13 | 64 | 128 | 4 | 1024 | 4 | 128 | 121.9666 | 51.1058 | **2.387x** | PASS |
| 14 | 32 | 1024 | 16 | 100000 | 2 | 1024 | N/A | N/A | N/A | CUDA OOM |

## Observations

- The final model was faster in cases 1, 5, 7, 11, and 13.
- The strongest measured improvement was case 13: **2.387x** at sequence length 1024.
- Case 11 also benefited substantially: **1.996x** with 16 attention heads, where GQA reduces K/V head work.
- Case 5 showed a **1.123x** gain at batch size 128.
- Small batches and short sequences generally favored the baseline because GQA, RoPE, and fused-GLU setup overhead dominated.
- Increasing model width to QKV/FFN dimension 1024 in case 8 was slower for the final model at this small sequence length.

## Resource failures

### Case 6

Batch size 10,000 with sequence length 128 and model width 128 requires roughly 5.12 GiB for the FP32 input tensor alone, before attention intermediates and model memory. The baseline then attempted additional allocations and exhausted the 9.64 GiB GPU.

### Case 14

Batch size 32, sequence length 100,000, and model width 1024 requires approximately 12.21 GiB for the FP32 input tensor alone. The benchmark failed while generating the input, before either model ran. Dense causal attention would additionally require an infeasible $O(S^2)$ attention workload.

## Conclusion

For the tested RTX 3080 workload, the composite model is most useful at larger sequence lengths, larger head counts, and higher batch sizes. Its best successful result was **2.387x** over baseline at case 13. Cases 6 and 14 are not executable on this GPU without reducing the batch/sequence dimensions or using a chunked/streaming benchmark with memory-efficient attention.
