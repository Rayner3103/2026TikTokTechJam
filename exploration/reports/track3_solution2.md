# Solution 2: Multi-Configuration Kernel with Auto-Tuning

**Importance: ████████░░ (8/10) — High value for adaptability**
**Ease of Implementation: ███████░░░ (7/10) — Medium complexity**

## Problem Solved

Addresses **Gap 2 (Benchmarking)** and **Gap 5 (Profiling)** — providing a flexible, auto-tuned kernel that adapts to different input shapes and achieves near-optimal performance across a range of configurations.

## Concept

Build a **configurable attention kernel** that automatically selects optimal parameters (tile sizes, block dimensions, number of warps) based on input shape and GPU characteristics. This provides robust performance across different use cases without manual tuning.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Auto-Tuned Attention Kernel                    │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────┐             │
│  │  Profiler       │  │  Tuner           │             │
│  │                 │  │                  │             │
│  │ • GPU specs     │  │ • Search space   │             │
│  │ • Input shape   │  │   configuration  │             │
│  │ • Memory info   │  │                  │             │
│  │ • Occupancy     │  │ • Benchmark      │             │
│  │   analysis      │  │   candidates     │             │
│  └────────┬────────┘  └────────┬─────────┘             │
│           │                    │                         │
│           └────────┬───────────┘                         │
│                    ▼                                      │
│  ┌─────────────────────────────────┐                   │
│  │  Kernel Selector               │                   │
│  │                                 │                   │
│  │ • Lookup cached results         │                   │
│  │ • Select optimal config         │                   │
│  │ • Handle edge cases             │                   │
│  └──────────────┬──────────────────┘                   │
│                 │                                       │
│                 ▼                                       │
│  ┌─────────────────────────────────┐                   │
│  │  Optimized Execution            │                   │
│  │                                 │                   │
│  │ • Selected kernel variant       │                   │
│  │ • Optimal tiling strategy       │                   │
│  │ • Best precision mode           │                   │
│  └─────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Technical Implementation

### 1. GPU Profiler
```python
class GPUProfiler:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.specs = self.gather_specs()
    
    def gather_specs(self) -> dict:
        """Gather GPU specifications"""
        props = torch.cuda.get_device_properties(self.device)
        return {
            'name': props.name,
            'total_memory': props.total_mem,
            'multiprocessor_count': props.multi_processor_count,
            'max_threads_per_block': props.max_threads_per_block,
            'max_warps_per_block': props.max_threads_per_block // 32,
            'shared_mem_per_block': props.shared_mem_per_block,
            'max_registers_per_block': props.regs_per_block,
            'warp_size': props.warp_size,
            'compute_capability': f'{props.major}.{props.minor}',
            'memory_bus_width': props.memory_bus_width,
            'memory_clock_rate': props.memory_clock_rate,
            'theoretical_bandwidth': (
                props.memory_clock_rate * 2e6 * 
                props.memory_bus_width / 8 / 1e9
            ),  # GB/s
        }
    
    def analyze_occupancy(self, block_size, shared_mem_per_thread=0):
        """Calculate GPU occupancy for given configuration"""
        max_threads = self.specs['max_threads_per_block']
        max_warps = self.specs['max_warps_per_block']
        shared_mem = self.specs['shared_mem_per_block']
        
        num_warps = block_size // 32
        blocks_per_sm = min(
            max_warps // num_warps,
            shared_mem // (shared_mem_per_thread * 32 * num_warps + 1)
            if shared_mem_per_thread > 0 else max_warps // num_warps,
        )
        
        occupancy = blocks_per_sm * num_warps / max_warps
        return {
            'blocks_per_sm': blocks_per_sm,
            'occupancy': occupancy,
            'active_warps': blocks_per_sm * num_warps,
        }
```

### 2. Configuration Search Space
```python
class ConfigSearchSpace:
    def __init__(self, head_dim, seq_len):
        self.head_dim = head_dim
        self.seq_len = seq_len
        
        # Define search space
        self.configs = []
        for block_m in [32, 64, 128, 256]:
            for block_n in [32, 64, 128]:
                for block_k in [16, 32, 64]:
                    for num_stages in [2, 3, 4]:
                        for num_warps in [2, 4, 8]:
                            # Filter invalid configurations
                            if block_k > head_dim:
                                continue
                            if block_n > seq_len:
                                continue
                            if block_m * block_n % (32 * 32) != 0:
                                continue  # Must be warp-aligned
                            
                            self.configs.append({
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_N': block_n,
                                'BLOCK_SIZE_K': block_k,
                                'num_stages': num_stages,
                                'num_warps': num_warps,
                            })
    
    def filter_by_gpu(self, gpu_specs: dict) -> list:
        """Filter configurations based on GPU capabilities"""
        valid = []
        max_regs = gpu_specs['max_registers_per_block']
        max_shared = gpu_specs['shared_mem_per_block']
        
        for config in self.configs:
            # Estimate register usage (heuristic)
            reg_estimate = config['BLOCK_SIZE_M'] * config['BLOCK_SIZE_K'] * 2
            if reg_estimate > max_regs:
                continue
            
            # Estimate shared memory usage
            shared_estimate = (
                config['BLOCK_SIZE_M'] * config['BLOCK_SIZE_K'] * 4 +  # Q tile
                config['BLOCK_SIZE_N'] * config['BLOCK_SIZE_K'] * 4 +  # K tile
                config['BLOCK_SIZE_N'] * config['HEAD_DIM'] * 4         # V tile
            )
            if shared_estimate > max_shared:
                continue
            
            valid.append(config)
        
        return valid
    
    def select_top_k(self, k=10):
        """Select top-k configurations based on heuristics"""
        # Heuristic scoring
        scored = []
        for config in self.configs:
            score = self.heuristic_score(config)
            scored.append((score, config))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
    
    def heuristic_score(self, config: dict) -> float:
        """Score a configuration based on heuristics"""
        score = 0.0
        
        # Prefer power-of-2 block sizes
        for size in [config['BLOCK_SIZE_M'], config['BLOCK_SIZE_N'], config['BLOCK_SIZE_K']]:
            if size & (size - 1) == 0:  # Power of 2
                score += 10
        
        # Prefer larger block sizes for throughput
        score += config['BLOCK_SIZE_M'] * config['BLOCK_SIZE_N'] / 1000
        
        # Prefer configurations that maximize occupancy
        occupancy = self.occupancy_score(config)
        score += occupancy * 100
        
        return score
```

### 3. Auto-Tuner
```python
class AutoTuner:
    def __init__(self, kernel_fn, benchmark_harness):
        self.kernel_fn = kernel_fn
        self.benchmark = benchmark_harness
        self.cache = {}  # Cache tuning results
    
    def tune(self, q, k, v, search_space: ConfigSearchSpace) -> dict:
        """Auto-tune the kernel for given inputs"""
        # Generate cache key
        cache_key = self.generate_cache_key(q, k, v)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get valid configurations
        gpu_profiler = GPUProfiler()
        valid_configs = search_space.filter_by_gpu(gpu_profiler.specs)
        
        # Benchmark each configuration
        results = []
        for config in valid_configs[:20]:  # Limit to top 20
            result = self.benchmark.benchmark_kernel(
                q, k, v,
                lambda q, k, v: self.kernel_fn(q, k, v, **config),
                f"config_{config}"
            )
            result['config'] = config
            results.append(result)
        
        # Select best configuration
        best = max(results, key=lambda r: r['throughput_tokens'])
        
        # Cache result
        self.cache[cache_key] = best
        
        return best
    
    def generate_cache_key(self, q, k, v) -> str:
        return f"b{q.size(0)}_s{q.size(1)}_h{q.size(2)}_d{q.size(3)}"
```

### 4. Configurable Attention Module
```python
class TunedAttention(torch.nn.Module):
    def __init__(self, num_heads, head_dim, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.tuner = AutoTuner(self.run_triton_kernel, AttentionBenchmark())
    
    def forward(self, q, k, v):
        # Determine search space
        search_space = ConfigSearchSpace(self.head_dim, q.size(1))
        
        # Tune for current input shape
        best_config = self.tuner.tune(q, k, v, search_space)
        
        # Run with best configuration
        return self.run_triton_kernel(
            q, k, v,
            BLOCK_SIZE_M=best_config['config']['BLOCK_SIZE_M'],
            BLOCK_SIZE_N=best_config['config']['BLOCK_SIZE_N'],
            BLOCK_SIZE_K=best_config['config']['BLOCK_SIZE_K'],
            num_stages=best_config['config']['num_stages'],
            num_warps=best_config['config']['num_warps'],
        )
```

## Justification

- **High Impact**: Auto-tuned kernels provide robust performance across different workloads
- **Medium Difficulty**: Combines existing Triton autotuning with custom search logic
- **Hackathon Feasible**: Search space can be limited to ensure tuning completes quickly
- **Evaluatable**: Clear metrics — auto-tuning time, performance across configurations, caching efficiency

## References

- Triton Autotuning: https://triton-lang.org/main/getting-started/tutorials/03-fused-matmul.html
- GPU Occupancy Calculator: https://developer.nvidia.com/content/cuda-occupancy-calculator
- Kernel Configuration Tuning: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#kernel-configuration
