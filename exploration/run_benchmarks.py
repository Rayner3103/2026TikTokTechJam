#!/usr/bin/env python3
"""
Comprehensive benchmark runner for Track 3 optimization directions.
Each direction runs via the standalone bench_single.py script.
"""
import sys, os, json, subprocess, time, signal

BASE = '/srv/vault/hot/SMU-Vault/Competition/2026TikTokTechJam/Track3'


def run_direction_subprocess(direction_num, direction_name, impl_path, strict_weights):
    """Run one direction in a fresh Python subprocess via bench_single.py."""
    cmd = [
        sys.executable,
        os.path.join(BASE, 'bench_single.py'),
        str(direction_num),
        impl_path,
        str(strict_weights),
    ]
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(
        cmd,
        cwd=BASE,
        env=env,
        capture_output=True, text=True, timeout=600
    )
    return result


def main():
    directions = [
        ('1', 'SDPA + torch.compile', f'{BASE}/direction1/direction1_optimized.py', True),
        ('2', 'QKV Fusion + SDPA', f'{BASE}/direction2/direction2_optimized.py', False),
        ('3', 'Layer Fusion + max-autotune', f'{BASE}/direction3/direction3_optimized.py', True),
    ]

    all_results = []
    for num, name, impl_path, strict in directions:
        print(f"\n{'='*70}")
        print(f">>> RUNNING DIRECTION {num}: {name}")
        print(f"{'='*70}")
        print(f"Impl: {impl_path}, strict: {strict}")

        r = run_direction_subprocess(num, name, impl_path, strict)

        print("\n--- STDOUT ---")
        print(r.stdout[-4000:] if r.stdout else "")
        if r.stderr:
            print("\n--- STDERR (last 2000 chars) ---")
            print(r.stderr[-2000:])

        if r.returncode != 0:
            print(f"\nERROR: exit code {r.returncode}")
            all_results.append(None)
            continue

        # Read back results
        rpath = os.path.join(BASE, f'direction{num}', 'benchmark_results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                res = json.load(f)
            all_results.append(res)
            print(f"\nResult for {num}: {name}")
            print(f"  Accuracy: {'PASS' if res['accuracy_passed'] else 'FAIL'}")
            print(f"  Speedup:  {res['speedup_factor']:.3f}x")
            print(f"  Max abs err: {res['max_abs_error']:.6g}")
            print(f"  Max rel err: {res['max_rel_error']:.6g}")
            print(f"  Baseline median: {res['baseline_median_ms']:.2f} ms")
            print(f"  Optimized median: {res['optimized_median_ms']:.2f} ms")
        else:
            all_results.append(None)

        # Small pause between directions
        time.sleep(1)

    # Combined results
    combined = {
        'config': {'batch_size': 2, 'seq_len': 2048, 'd_model': 512,
                    'num_heads': 8, 'ffn_dim': 2048, 'num_layers': 12, 'causal': True},
        'results': all_results
    }
    with open(f'{BASE}/benchmark_results.json', 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved: {BASE}/benchmark_results.json")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Direction':<35} {'Accuracy':<10} {'Speedup':<10} {'Max Abs Err':<15}")
    print("-" * 70)
    for i, r in enumerate(all_results):
        if r:
            st = "PASS" if r['accuracy_passed'] else "FAIL"
            print(f"{r['direction']:<35} {st:<10} {r['speedup_factor']:.2f}x    {r['max_abs_error']:.2e}")


if __name__ == "__main__":
    main()
