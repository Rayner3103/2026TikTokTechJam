"""Benchmark the composite final model against the reference baseline.

The composite changes the parameter topology (GQA and fused GLU), so baseline
checkpoint copying is intentionally not used. Both models are independently
initialized and compared on identical inputs for latency, throughput, shape,
and finite-output checks.
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import torch

import final_model
from torch_transformer_benchmark import (
    BaselineTransformer,
    generate_random_case,
    resolve_device,
    resolve_dtype,
)


def measure(model, x, mask, device, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                model(x, mask)
                end.record()
            torch.cuda.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            samples = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                model(x, mask)
                samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples), statistics.fmean(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")
    parser.add_argument("--routing", action="store_true", help="Enable Phase 3 adaptive routing")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    config = SimpleNamespace(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    if args.heads % args.kv_heads != 0:
        raise ValueError("--kv-heads must divide --heads")
    config.num_kv_heads = args.kv_heads
    config.max_seq_len = max(2048, args.seq_len)
    config.routing_enabled = args.routing

    torch.manual_seed(1234)
    baseline = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    optimized = final_model.OptimizedTransformer(config).to(device=device, dtype=dtype).eval()
    if args.compile:
        baseline = torch.compile(baseline, mode=args.compile_mode)
        optimized = torch.compile(optimized, mode=args.compile_mode)

    x, mask = generate_random_case(config, device, dtype, 1234, args.padding_ratio, 1.0)
    with torch.inference_mode():
        baseline_output = baseline(x, mask)
        optimized_output = optimized(x, mask)
    shape_ok = baseline_output.shape == optimized_output.shape == x.shape
    finite_ok = bool(torch.isfinite(optimized_output).all().item())

    baseline_median, baseline_mean = measure(baseline, x, mask, device, args.warmup, args.repeats)
    optimized_median, optimized_mean = measure(optimized, x, mask, device, args.warmup, args.repeats)
    speedup = baseline_median / optimized_median
    tokens = args.batch_size * args.seq_len

    print("=== Final composite benchmark ===")
    print(f"config: batch={args.batch_size}, seq_len={args.seq_len}, d_model={args.d_model}, heads={args.heads}, kv_heads={args.kv_heads}, layers={args.layers}, dtype={dtype}, device={device}")
    print("phase 1: SDPA" + (" + torch.compile" if args.compile else ""))
    print("phase 2: GQA + fused SiLU GLU")
    print(f"phase 3: adaptive routing={'enabled' if args.routing else 'disabled (training-ready)'}")
    print(f"output_contract: {'PASS' if shape_ok else 'FAIL'} | finite_output: {'PASS' if finite_ok else 'FAIL'}")
    print(f"baseline : median={baseline_median:.4f} ms | mean={baseline_mean:.4f} ms | throughput={tokens * 1000 / baseline_median:.2f} token/s")
    print(f"optimized: median={optimized_median:.4f} ms | mean={optimized_mean:.4f} ms | throughput={tokens * 1000 / optimized_median:.2f} token/s")
    print(f"speedup  : {speedup:.3f}x based on median latency")
    print("accuracy : NOT COMPARED because GQA/GLU change the architecture; use task-level validation after checkpoint training/conversion")
    return 0 if shape_ok and finite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
