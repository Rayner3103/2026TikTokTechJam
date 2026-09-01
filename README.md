# Rayner Transformer Exploration

This folder packages Rayner's transformer optimization exploration so another
developer can clone the project, reproduce the terminal experiments, and find
the resulting reports in one place.

## Layout

```text
Rayner/
├── run_final_sweep.py              # one command for the final multi-config sweep
├── REPORT.md                       # generated final-model benchmark report
├── FINAL_BENCHMARK.md              # recorded root final-model benchmark
├── final_model.py                  # final composite model
├── final_model_benchmark.py        # baseline comparison harness
├── torch_transformer_benchmark.py  # shared baseline/reference implementation
└── exploration/
│   ├── directions/                 # Directions 1-7
│   ├── reports/                    # research notes and previous reports
│   └── results/                    # captured exploration JSON artifacts
```

## Requirements

- Linux with an NVIDIA CUDA GPU for the reported timings.
- Python 3.10 or newer.
- PyTorch 2.x with CUDA support.
- `flash-attn` is not required by the Rayner composite harness.

Create an environment matching the tested CUDA stack:

```bash
conda create -n rayner-transformer python=3.10 -y
conda activate rayner-transformer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Check the GPU before benchmarking:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Reproduce the final benchmark

From this directory:

```bash
cd Rayner
python run_final_sweep.py
```

The runner starts one fresh Python process per configuration. It prints each
baseline/final comparison to the terminal, writes `REPORT.md`, and saves raw
output in `run_config_1.log` through `run_config_14.log`. Cases 6 and 14 are
expected to exceed a 9.64 GiB RTX 3080 because their requested tensors and
dense attention intermediates are too large; they remain documented rather
than being skipped.

Run one smaller smoke test directly:

```bash
cd Rayner
python final_model_benchmark.py --batch-size 8 --seq-len 512 --d-model 128 \
  --heads 8 --kv-heads 2 --ffn-dim 512 --layers 4 --causal \
  --warmup 10 --repeats 30
```

The complete recorded sweep is in [FINAL_BENCHMARK.md](FINAL_BENCHMARK.md).
The latest generated multi-process sweep is in [REPORT.md](REPORT.md).

The Direction 1-7 exploration results are summarized in
[exploration/reports/EXPLORATION_BENCHMARK.md](exploration/reports/EXPLORATION_BENCHMARK.md).

## Interpretation

The composite model changes the architecture through GQA and fused GLU, so a
baseline checkpoint cannot be copied into it and used as an elementwise
accuracy oracle. The final sweep therefore reports performance plus output
contract/finite checks. Task-level validation is required after training or
checkpoint conversion.