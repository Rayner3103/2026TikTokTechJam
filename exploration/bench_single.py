#!/usr/bin/env python3
"""
Standalone benchmark for ONE direction.
Usage: python3 bench_single.py <direction_num> <impl_module_path> <use_compile> <strict_weights>

Runs in a fresh process to avoid GPU memory leaks from torch.compile.
"""
import sys, json, statistics, copy, os, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

direction_num = sys.argv[1]
impl_path = sys.argv[2]
use_compile = sys.argv[3].lower() == "true"
strict_weights = sys.argv[4].lower() == "true"

# Import the optimized model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"direction{direction_num}"))
mod_name = os.path.splitext(os.path.basename(impl_path))[0]
spec = __import__("importlib.util").util.spec_from_file_location(mod_name, impl_path)
mod = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mod)
OptimizedTransformer = mod.OptimizedTransformer

from torch_transformer_benchmark import (
    TransformerConfig, BaselineTransformer, copy_model_weights,
    resolve_device, resolve_dtype, generate_random_case,
    compare_outputs, warmup_model, benchmark_once,
)

device = resolve_device("auto")
dtype = resolve_dtype("float32")

config = TransformerConfig(
    batch_size=2, seq_len=2048, d_model=512,
    num_heads=8, ffn_dim=2048, num_layers=12, causal=True,
)

print(f"Config: batch={config.batch_size}, seq={config.seq_len}, d_model={config.d_model}, layers={config.num_layers}", flush=True)
print(f"Device: {device}, dtype: {dtype}", flush=True)
print(f"GPU mem before: {torch.cuda.memory_allocated()/1e6:.0f} MB", flush=True)

baseline = BaselineTransformer(config).to(device).to(dtype)
baseline.eval()

opt = OptimizedTransformer(config).to(device).to(dtype)
copy_model_weights(baseline, opt, strict=strict_weights)
opt.eval()

if use_compile:
    print("Applying torch.compile(reduce-overhead)...", flush=True)
    opt = torch.compile(opt, mode="reduce-overhead", fullgraph=False)
    x_tmp, m_tmp = generate_random_case(config, device, dtype, seed=9999, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        opt(x_tmp, m_tmp)
        torch.cuda.synchronize(device)
    print("torch.compile warmup done.", flush=True)

print(f"GPU mem after compile: {torch.cuda.memory_allocated()/1e6:.0f} MB", flush=True)

# ACCURACY
all_passed = True
g_max_abs = 0.0
g_max_rel = 0.0
total_failed = 0
total_el = 0
failures = []

with torch.inference_mode():
    for trial in range(20):
        x, vm = generate_random_case(config, device, dtype, seed=trial, padding_ratio=0.0, input_scale=1.0)
        ref = baseline(x, vm)
        c = opt(x, vm)
        r = compare_outputs(ref, c, rtol=0.01, atol=0.001)
        all_passed &= r.passed
        g_max_abs = max(g_max_abs, r.max_abs_error)
        g_max_rel = max(g_max_rel, r.max_relative_error)
        total_failed += r.failed_elements
        total_el += r.total_elements
        st = "PASS" if r.passed else "FAIL"
        print(f"  trial {trial+1:02d}: {st} | max_abs={r.max_abs_error:.6g} | max_rel={r.max_relative_error:.6g}", flush=True)
        if not r.passed:
            failures.append({"trial": trial, "failed_elements": r.failed_elements,
                             "max_abs": r.max_abs_error, "max_rel": r.max_relative_error,
                             "worst_index": list(r.worst_index),
                             "failed_features": r.failed_feature_dims[:10]})

print(f"Accuracy: {'PASS' if all_passed else 'FAIL'} | max_abs={g_max_abs:.6g} | max_rel={g_max_rel:.6g}", flush=True)

# TIMING
x_t, vm_t = generate_random_case(config, device, dtype, seed=1000, padding_ratio=0.0, input_scale=1.0)

warmup_model(baseline, x_t, vm_t, 10, device)
b_samples = benchmark_once(baseline, x_t, vm_t, 50, device)

warmup_model(opt, x_t, vm_t, 10, device)
o_samples = benchmark_once(opt, x_t, vm_t, 50, device)

b_med = statistics.median(b_samples)
o_med = statistics.median(o_samples)
speedup = b_med / o_med if o_med > 0 else float('inf')

print(f"Baseline:   mean={statistics.fmean(b_samples):.2f}ms, median={b_med:.2f}ms", flush=True)
print(f"Optimized:  mean={statistics.fmean(o_samples):.2f}ms, median={o_med:.2f}ms", flush=True)
print(f"Speedup:    {speedup:.3f}x (median)", flush=True)

peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0

result = {
    "direction": f"Direction {direction_num}",
    "accuracy_passed": all_passed,
    "max_abs_error": g_max_abs,
    "max_rel_error": g_max_rel,
    "failed_elements": total_failed,
    "total_elements": total_el,
    "baseline_mean_ms": statistics.fmean(b_samples),
    "baseline_median_ms": b_med,
    "optimized_mean_ms": statistics.fmean(o_samples),
    "optimized_median_ms": o_med,
    "speedup_factor": speedup,
    "peak_cuda_memory_mb": float(peak_mem),
    "failures": failures,
}

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"direction{direction_num}")
with open(os.path.join(outdir, "benchmark_results.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to direction{direction_num}/benchmark_results.json", flush=True)
