#!/usr/bin/env python3
"""
Unified benchmark runner for Track 3.
Auto-discovers all direction*/ folders, benchmarks each sequentially.
Usage: python3 run_all.py [--start N --end M]
"""
import sys, os, json, glob, argparse, subprocess
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent
DIRECTIONS_DIR = TRACK_DIR / "directions"
RESULTS_FILE = TRACK_DIR / "results.json"


def discover_directions(start=None, end=None):
    """Auto-discover all direction*/ folders."""
    dirs = sorted(glob.glob(str(DIRECTIONS_DIR / "direction[0-9]*")))
    filtered = []
    for d in dirs:
        n = int(os.path.basename(d).replace("direction", ""))
        if start is not None and n < start:
            continue
        if end is not None and n > end:
            continue
        filtered.append((n, d))
    return filtered


def run_direction_subprocess(direction_num, impl_path, strict):
    """Run ONE direction in a subprocess, return results dict."""
    # Build import path: e.g. direction1_optimized -> from direction1_optimized import Transformer
    impl_name = Path(impl_path).stem  # e.g. "direction1_optimized"
    # Properly escape path for use in f-string
    impl_path_escaped = impl_path.replace("\\", "\\\\")

    script = f"""
import sys, os, importlib.util, json, time, numpy as np
import torch

# Load the implementation module
spec = importlib.util.spec_from_file_location("{impl_name}", "{impl_path_escaped}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Transformer = mod.OptimizedTransformer

# --- Helper: run benchmark with multiple iterations for latency stats ---
def benchmark_model(model, x, valid_mask, num_iters=10, warmup=3):
    \"\"\"Run multiple iterations and return latency statistics.\"\"\"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, valid_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iters):
        t0 = time.time()
        with torch.no_grad():
            _ = model(x, valid_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000  # ms
        times.append(elapsed)
    
    times = np.array(times)
    return {{
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }}

# --- Config variations to test ---
CONFIGS = [
    {{"name": "small", "batch": 4, "seq_len": 128, "d_model": 64, "num_heads": 4, "num_layers": 2}},
    {{"name": "medium", "batch": 8, "seq_len": 512, "d_model": 128, "num_heads": 8, "num_layers": 4}},
    {{"name": "large", "batch": 16, "seq_len": 1024, "d_model": 256, "num_heads": 8, "num_layers": 6}},
]

STRICT = {strict}
USE_COMPUTE = True
NUM_HEADS = 8
FFN_DIM = 512
VOCAB_SIZE = 10000

# Create config object
class Config:
    def __init__(self, batch_size, seq_len, d_model, num_heads, num_layers):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = FFN_DIM
        self.vocab_size = VOCAB_SIZE
        self.causal = True
        self.compile = False

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEBUG: Using device: {{device}}", file=sys.stderr)
print(f"DEBUG: CUDA available: {{torch.cuda.is_available()}}", file=sys.stderr)

all_results = []

# Test each configuration
for cfg in CONFIGS:
    cfg_name = cfg["name"]
    batch_size = cfg["batch"]
    seq_len = cfg["seq_len"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]
    
    config = Config(batch_size, seq_len, d_model, num_heads, num_layers)
    
    # Dummy data - embeddings, not token indices
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # --- Baseline (no compile) ---
    model = Transformer(config)
    model = model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    base_stats = benchmark_model(model, x, valid_mask)
    base_peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # --- Optimized (with compile) ---
    config_opt = Config(batch_size, seq_len, d_model, num_heads, num_layers)
    config_opt.compile = USE_COMPUTE
    opt_model = Transformer(config_opt)
    opt_model = opt_model.to(device)
    opt_model.eval()
    
    # Copy weights from baseline to optimized for fair comparison
    try:
        opt_model.load_state_dict(model.state_dict())
    except RuntimeError:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    opt_stats = benchmark_model(opt_model, x, valid_mask)
    opt_peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # --- Accuracy check ---
    with torch.no_grad():
        y_base = model(x, valid_mask)
        y_opt = opt_model(x, valid_mask)
        y_base2 = model(x, valid_mask)
    abs_err = (abs(y_base - y_opt)).mean().item()
    rel_err = (abs(y_base - y_opt) / (abs(y_base2) + 1e-8)).mean().item()
    acc_pass = abs_err < 0.002 or rel_err < 0.02
    
    # --- Throughput (tokens/sec) ---
    base_throughput = (batch_size * seq_len) / (base_stats["mean"] / 1000)  # tokens/sec
    opt_throughput = (batch_size * seq_len) / (opt_stats["mean"] / 1000)
    
    cfg_result = {{
        "config": cfg_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "baseline": {{
            "latency_ms": {{"mean": round(base_stats["mean"], 3), "median": round(base_stats["median"], 3), 
                           "std": round(base_stats["std"], 3), "p95": round(base_stats["p95"], 3), "p99": round(base_stats["p99"], 3)}},
            "throughput_tokens_sec": round(base_throughput, 1),
            "peak_memory_mb": round(base_peak_mem, 1),
        }},
        "optimized": {{
            "latency_ms": {{"mean": round(opt_stats["mean"], 3), "median": round(opt_stats["median"], 3), 
                           "std": round(opt_stats["std"], 3), "p95": round(opt_stats["p95"], 3), "p99": round(opt_stats["p99"], 3)}},
            "throughput_tokens_sec": round(opt_throughput, 1),
            "peak_memory_mb": round(opt_peak_mem, 1),
        }},
        "speedup": round(base_stats["mean"] / opt_stats["mean"], 2) if opt_stats["mean"] > 0 else 0,
        "throughput_gain": round((opt_throughput / base_throughput - 1) * 100, 1),
        "memory_savings_pct": round((1 - opt_peak_mem / base_peak_mem) * 100, 1) if base_peak_mem > 0 else 0,
        "abs_error": round(abs_err, 6),
        "rel_error": round(rel_err, 6),
        "accuracy_pass": acc_pass,
    }}
    all_results.append(cfg_result)

print(json.dumps({{
    "direction": {direction_num},
    "impl": "{impl_path_escaped}",
    "device": str(device),
    "cuda_available": torch.cuda.is_available(),
    "configs": all_results,
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        return {
            "direction": direction_num,
            "impl": impl_path,
            "error": result.stderr[:500],
        }
    return json.loads(result.stdout.strip())


def run_validation_on_direction(direction_num, impl_path):
    """Run all 5 validation test cases on a specific direction implementation."""
    
    test_cases = [
        ("causal_padded", "--causal --padding-ratio 0.25 --accuracy-trials 5 --warmup 3 --repeats 15 --benchmark-rounds 1"),
        ("float16_causal", "--causal --dtype float16 --accuracy-trials 5 --warmup 3 --repeats 15 --benchmark-rounds 1"),
        ("bfloat16_padded", "--dtype bfloat16 --padding-ratio 0.25 --accuracy-trials 5 --warmup 3 --repeats 15 --benchmark-rounds 1"),
        ("minimal_sequence", "--batch-size 2 --seq-len 1 --d-model 64 --heads 8 --ffn-dim 128 --layers 1 --padding-ratio 0.5 --accuracy-trials 5 --warmup 2 --repeats 10 --benchmark-rounds 1"),
        ("compiled", "--batch-size 2 --seq-len 64 --d-model 128 --heads 8 --ffn-dim 512 --layers 2 --causal --compile-user --compile-mode reduce-overhead --accuracy-trials 3 --warmup 3 --repeats 10 --benchmark-rounds 1"),
    ]
    
    validation_results = []
    impl_name = Path(impl_path).stem
    impl_path_escaped = impl_path.replace("\\", "\\\\")
    
    for test_name, args_str in test_cases:
        # Create wrapper that loads direction and patches torch_transformer_benchmark
        validation_script = f"""
import sys, os, importlib.util, argparse
sys.path.insert(0, os.path.dirname("{impl_path_escaped}"))
sys.path.insert(0, "{TRACK_DIR}")

# First import torch_transformer_benchmark
import torch_transformer_benchmark as tb
import torch

# Load the direction's implementation
spec = importlib.util.spec_from_file_location("{impl_name}", "{impl_path_escaped}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Replace the UserOptimizedTransformer class with direction's OptimizedTransformer
tb.UserOptimizedTransformer = mod.OptimizedTransformer

# Parse args and run benchmark
sys.argv = ["benchmark"] + {repr(args_str.split())} + ["--strict-weights", "false"]
sys.argv = ["benchmark"] + {repr(args_str.split())} + ["--non-strict-weight-copy"]
exit_code = tb.main()
"""
        
        result = subprocess.run(
            [sys.executable, "-c", validation_script],
            capture_output=True, text=True, cwd=TRACK_DIR, timeout=300
        )
        
        output = result.stdout + result.stderr
        accuracy_passed = "PASS" in output and "summary: PASS" in output
        
        # Extract speedup
        speedup = 1.0
        for line in output.split('\n'):
            if 'speedup' in line.lower() and 'x' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        speedup_str = parts[-1].strip().split('x')[0].strip()
                        speedup = float(speedup_str)
                except (ValueError, IndexError):
                    pass
        
        # Extract latencies
        baseline_median = None
        optimized_median = None
        for line in output.split('\n'):
            if 'baseline' in line.lower() and 'median' in line:
                try:
                    median_str = line.split('median=')[1].split()[0]
                    baseline_median = float(median_str)
                except (ValueError, IndexError):
                    pass
            elif 'optimized' in line.lower() and 'median' in line:
                try:
                    median_str = line.split('median=')[1].split()[0]
                    optimized_median = float(median_str)
                except (ValueError, IndexError):
                    pass
        
        validation_results.append({
            "test": test_name,
            "accuracy_passed": accuracy_passed,
            "speedup": round(speedup, 3),
            "baseline_ms": baseline_median,
            "optimized_ms": optimized_median,
        })
    
    return validation_results


def run_validation_tests():
    """Run comprehensive validation tests on torch_transformer_benchmark.py baseline."""
    print(f"\n{'='*60}")
    print("VALIDATION: Running torch_transformer_benchmark.py tests")
    print(f"{'='*60}\n")
    
    test_cases = [
        {
            "name": "causal_padded",
            "args": "--causal --padding-ratio 0.25 --accuracy-trials 10 --warmup 5 --repeats 30 --benchmark-rounds 2"
        },
        {
            "name": "float16_causal",
            "args": "--causal --dtype float16 --accuracy-trials 10 --warmup 5 --repeats 30 --benchmark-rounds 2"
        },
        {
            "name": "bfloat16_padded",
            "args": "--dtype bfloat16 --padding-ratio 0.25 --accuracy-trials 10 --warmup 5 --repeats 30 --benchmark-rounds 2"
        },
        {
            "name": "minimal_sequence",
            "args": "--batch-size 2 --seq-len 1 --d-model 64 --heads 8 --ffn-dim 128 --layers 1 --padding-ratio 0.5 --accuracy-trials 10 --warmup 3 --repeats 20 --benchmark-rounds 2"
        },
        {
            "name": "compiled_user",
            "args": "--batch-size 2 --seq-len 64 --d-model 128 --heads 8 --ffn-dim 512 --layers 2 --causal --compile-user --compile-mode reduce-overhead --accuracy-trials 5 --warmup 5 --repeats 20 --benchmark-rounds 2"
        }
    ]
    
    validation_results = []
    
    for i, test in enumerate(test_cases, 1):
        test_name = test["name"]
        cmd = f"python3 {TRACK_DIR / 'torch_transformer_benchmark.py'} {test['args']}"
        
        print(f"[{i}/{len(test_cases)}] Running validation: {test_name}...")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=TRACK_DIR, timeout=600)
        
        # Parse output for key metrics
        output = result.stdout + result.stderr
        accuracy_passed = "PASS" in output and "summary: PASS" in output
        
        # Extract speedup if present
        speedup = 1.0
        for line in output.split('\n'):
            if 'speedup' in line.lower() and 'x' in line:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        speedup_str = parts[-1].strip().split('x')[0].strip()
                        speedup = float(speedup_str)
                except (ValueError, IndexError):
                    pass
        
        # Extract baseline and optimized latencies
        baseline_median = None
        optimized_median = None
        for line in output.split('\n'):
            if 'baseline' in line.lower() and 'median' in line:
                try:
                    median_str = line.split('median=')[1].split()[0]
                    baseline_median = float(median_str)
                except (ValueError, IndexError):
                    pass
            elif 'optimized' in line.lower() and 'median' in line:
                try:
                    median_str = line.split('median=')[1].split()[0]
                    optimized_median = float(median_str)
                except (ValueError, IndexError):
                    pass
        
        validation_results.append({
            "test_name": test_name,
            "accuracy_passed": accuracy_passed,
            "speedup": round(speedup, 3),
            "baseline_median_ms": baseline_median,
            "optimized_median_ms": optimized_median,
            "exit_code": result.returncode,
        })
        
        status = "✓ PASS" if accuracy_passed else "✗ FAIL"
        print(f"  {status} | Speedup: {speedup:.3f}x")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Track 3 unified benchmark runner")
    parser.add_argument("--start", type=int, help="Start direction number (inclusive)")
    parser.add_argument("--end", type=int, help="End direction number (inclusive)")
    args = parser.parse_args()

    directions = discover_directions(args.start, args.end)
    if not directions:
        print("No direction*/ folders found!")
        sys.exit(1)

    print(f"Discovered {len(directions)} direction(s): {[d[0] for d in directions]}")
    results = []
    strict_weights = True  # Set False if compile breaks on weight mismatches

    for i, (num, dirpath) in enumerate(directions):
        impl_path = os.path.join(dirpath, f"direction{num}_optimized.py")
        if not os.path.exists(impl_path):
            print(f"[{i+1}/{len(directions)}] No {impl_path}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(directions)}] Running direction {num}...")
        print(f"{'='*60}")

        result = run_direction_subprocess(num, impl_path, strict_weights)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error'][:200]}")
        else:
            # New complex results structure with multiple configs
            if "configs" in result:
                for cfg in result["configs"]:
                    print(f"\n  Config: {cfg['config']} ({cfg['batch_size']}B x {cfg['seq_len']}L)")
                    print(f"    Baseline:  {cfg['baseline']['latency_ms']['mean']:.3f}ms (p95: {cfg['baseline']['latency_ms']['p95']:.3f}ms) | Throughput: {cfg['baseline']['throughput_tokens_sec']:.0f} tok/s | Memory: {cfg['baseline']['peak_memory_mb']:.1f}MB")
                    print(f"    Optimized: {cfg['optimized']['latency_ms']['mean']:.3f}ms (p95: {cfg['optimized']['latency_ms']['p95']:.3f}ms) | Throughput: {cfg['optimized']['throughput_tokens_sec']:.0f} tok/s | Memory: {cfg['optimized']['peak_memory_mb']:.1f}MB")
                    print(f"    Speedup: {cfg['speedup']}x | Throughput gain: {cfg['throughput_gain']:+.1f}% | Memory saved: {cfg['memory_savings_pct']:+.1f}% | Accuracy: {cfg['accuracy_pass']}")
            else:
                # Fallback to old format for backward compatibility
                print(f"  Baseline: {result['base_ms_per_token']} ms/token")
                print(f"  Optimized: {result['opt_ms_per_token']} ms/token")
                print(f"  Speedup: {result['speedup']}x")
                print(f"  Abs error: {result['abs_error']}")
                print(f"  Accuracy pass: {result['accuracy_pass']}")
                print(f"  GPU memory: {result.get('gpu_memory_mb', 'N/A')} MB")

        # Run validation tests on this direction
        print(f"\n  Running validation tests on Direction {num}...")
        direction_validation = run_validation_on_direction(num, impl_path)
        if "configs" in result:
            result["validation_tests"] = direction_validation
        
        # Print validation results
        for val in direction_validation:
            status = "✓" if val["accuracy_passed"] else "✗"
            speedup_str = f"{val['speedup']:.3f}x" if val["speedup"] else "N/A"
            print(f"    {status} {val['test']:<20} | Speedup: {speedup_str:>8} | Accuracy: {val['accuracy_passed']}")

    # Run validation tests on baseline
    validation_results = run_validation_tests()
    
    # Add validation results to output
    results.append({
        "baseline_validation_tests": validation_results
    })

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    passed = 0
    failed = 0
    for r in results:
        if "error" in r:
            failed += 1
        elif "configs" in r:
            # For new complex results, count accuracy passes
            for cfg in r["configs"]:
                if cfg.get("accuracy_pass"):
                    passed += 1
                else:
                    failed += 1
        else:
            # For old format
            if r.get("accuracy_pass"):
                passed += 1
            else:
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
