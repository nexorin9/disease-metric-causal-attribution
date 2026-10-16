"""核心算法包（spec.md §9-§10 因果推理管线）。

职责：
- 业务 DAG 模板（dag_builder）
- Pydantic 节点 / 边 / 效应 / 反驳 / 反事实 / 审计 schema（node_schema）
- 因果流水线（CausalPipeline：model → identify → estimate → refute）
- 反事实模拟（DoSampler / BatchDoSampler）
- 反驳多档（RefuterPipeline）
- 对照基线（CorrelationBaseline）
- 样本量守门（SampleSizeGuard）
- 估计器选择器（EstimatorSelector）

不依赖 FastAPI / 业务存储（db / drafts 模块）；与 adapters 仅通过数据接口耦合。

源产品能力参考：see GITHUB_CODE_REF.md 与 spec.md §9-§10（四步流水线 / DoSampler /
GraphRefuter / gcm/equation_parser 与业务融合）；README 不得写源 repo 名。
"""
from __future__ import annotations

__all__ = [
    "baseline",
    "dag_builder",
    "dosampler",
    "dosampler_batch",
    "equation_parser",
    "equation_templates",
    "estimator_selector",
    "exception_handlers",
    "graph_refuter",
    "node_schema",
    "path_traverse",
    "path_traverse_extension",
    "pipeline",
    "refuter",
    "sample_size_gate",
]
