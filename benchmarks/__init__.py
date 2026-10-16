"""benchmarks 包入口（spec.md §11 第 5 步 / T18）。

包含性能与稳定性实测 benchmark：

- ``run_perf_benchmark.py``：跑 48 个 (病种 × outcome × 样本量) 组合并落盘
  ``perf_records/benchmark_*.json`` 与 ``perf_records/benchmark_summary.json``
- ``test_perf_bench.py``：pytest 验收层，断言 48 个 JSON 落盘 +
  n=2000 < 60s + error_rate < 5%
- ``run_parallel.sh``：一键跑全 benchmark（PYTHONPATH=. 引导）
"""
