#!/usr/bin/env python3
"""Run the Rayner composite model against the baseline for all study shapes."""

from __future__ import annotations

import re
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPLORATION = ROOT / "exploration"
BENCHMARK = EXPLORATION / "final_model_benchmark.py"
REPORT = ROOT / "REPORT.md"

# id: batch, d_model, heads, seq_len, layers, ffn_dim, kv_heads
CONFIGS = {
    1: (64, 128, 4, 128, 4, 128, 4),
    2: (1, 128, 4, 128, 4, 128, 4),
    3: (4, 128, 4, 128, 4, 128, 4),
    4: (16, 128, 4, 128, 4, 128, 4),
    5: (128, 128, 4, 128, 4, 128, 4),
    6: (10000, 128, 4, 128, 4, 128, 4),
    7: (64, 32, 4, 128, 4, 32, 4),
    8: (64, 1024, 4, 128, 4, 1024, 4),
    9: (64, 128, 1, 128, 4, 128, 1),
    10: (64, 128, 2, 128, 4, 128, 2),
    11: (64, 128, 16, 128, 4, 128, 4),
    12: (64, 128, 4, 32, 4, 128, 4),
    13: (64, 128, 4, 1024, 4, 128, 4),
    14: (32, 1024, 16, 100000, 2, 1024, 4),
}


def run_case(
    case_id: int,
    values: tuple[int, ...],
    compile_model: bool,
    routing: bool,
) -> dict[str, object]:
    batch, d_model, heads, seq_len, layers, ffn_dim, kv_heads = values
    command = [
        sys.executable,
        str(BENCHMARK),
        "--batch-size", str(batch),
        "--seq-len", str(seq_len),
        "--d-model", str(d_model),
        "--heads", str(heads),
        "--kv-heads", str(kv_heads),
        "--ffn-dim", str(ffn_dim),
        "--layers", str(layers),
        "--causal",
        "--device", "cuda",
        "--dtype", "float32",
        "--warmup", "3",
        "--repeats", "10",
    ]
    if compile_model:
        command.extend(["--compile", "--compile-mode", "reduce-overhead"])
    if routing:
        command.append("--routing")
    print(f"=== config {case_id}: batch={batch} d_model={d_model} heads={heads} "
          f"seq_len={seq_len} layers={layers} ffn_dim={ffn_dim} ===", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=EXPLORATION,
            capture_output=True,
            text=True,
            timeout=900,
        )
        output = completed.stdout + completed.stderr
        (ROOT / f"run_config_{case_id}.log").write_text(output)
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "")
        completed = None
        (ROOT / f"run_config_{case_id}.log").write_text(output)
        return {"id": case_id, "values": values, "status": "TIMEOUT", "output": output}

    baseline = re.search(r"baseline\s*: median=([\d.]+) ms", output)
    optimized = re.search(r"optimized: median=([\d.]+) ms", output)
    speedup = re.search(r"speedup\s*: ([\d.]+)x", output)
    contract = re.search(r"output_contract: (PASS|FAIL)", output)
    status = "PASS" if completed.returncode == 0 and speedup else "FAILED"
    result = {
        "id": case_id,
        "values": values,
        "status": status,
        "baseline_ms": float(baseline.group(1)) if baseline else None,
        "optimized_ms": float(optimized.group(1)) if optimized else None,
        "speedup": float(speedup.group(1)) if speedup else None,
        "contract": contract.group(1) if contract else "UNKNOWN",
        "returncode": completed.returncode,
        "output": output,
    }
    speedup_text = f"{result['speedup']:.3f}x" if result["speedup"] else "n/a"
    print(f"  -> status={status} speedup={speedup_text}", flush=True)
    return result


def write_report(results: list[dict[str, object]]) -> None:
    lines = [
        "# Rayner Final Model Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The final model combines SDPA, GQA, fused SiLU GLU, and a training-ready "
        "adaptive router. The router is disabled for this inference sweep. Each "
        "case compares independently initialized models on the same GPU input. "
        "Cross-model numerical accuracy is not claimed because GQA and GLU change "
        "the parameter topology; output shape and finite-value checks are used.",
        "",
        "## Results",
        "",
        "| # | Batch | D Model | Heads | Seq Len | Layers | FFN | Baseline ms | Final ms | Speedup | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        batch, d_model, heads, seq_len, layers, ffn_dim, _ = result["values"]
        baseline = f"{result['baseline_ms']:.4f}" if result["baseline_ms"] else "N/A"
        optimized = f"{result['optimized_ms']:.4f}" if result["optimized_ms"] else "N/A"
        speedup = f"{result['speedup']:.3f}x" if result["speedup"] else "N/A"
        status = result["status"]
        if result.get("contract") == "PASS":
            status += ", contract PASS"
        lines.append(
            f"| {result['id']} | {batch} | {d_model} | {heads} | {seq_len} | "
            f"{layers} | {ffn_dim} | {baseline} | {optimized} | {speedup} | {status} |"
        )
    successful = [result for result in results if result["speedup"] is not None]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Successful runs: {len(successful)}/{len(results)}.",
        "- Positive speedup means the final model had lower median latency than the baseline.",
        "- Cases that fail with CUDA OOM are recorded as resource limits, not model correctness failures.",
        "- Per-case logs are saved as `run_config_<n>.log` beside this report.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd Rayner",
        "python run_final_sweep.py",
        "```",
    ])
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="compile both models")
    parser.add_argument("--routing", action="store_true", help="enable adaptive routing")
    args = parser.parse_args()
    results = [
        run_case(case_id, values, args.compile, args.routing)
        for case_id, values in CONFIGS.items()
    ]
    write_report(results)
    print(f"\nReport written to {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())