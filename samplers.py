# samplers.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ir_methods

@dataclass(frozen=True)
class SamplerContext:
    """Shared context needed to instantiate samplers."""
    cluster_label_model: Optional[tuple] = None    # (centroids, label_distributions)
    n_samples: int = 1000


def make_sampler(kind: str, ctx: SamplerContext, **extra_kwargs) -> Dict[str, Any]:
    """
    Returns a sampler dict of the form:
        {"fn": <callable>, "args": {...}}
    which matches the interface expected by training.optimize_vrisk.
    """
    k = kind.lower()
    ## Independent disagreement
    if k == "cluster_label_dist":
        if ctx.cluster_label_model is None:
            raise ValueError("cluster_label_dist sampler requires cluster_label_model in SamplerContext.")
        centroids, label_distributions = ctx.cluster_label_model
        return {
            "fn": ir_methods.sample_cluster_label_dist,
            "args": {
                "n_samples": ctx.n_samples,
                "centroids": centroids,
                "cluster_label_dists": label_distributions,
                **extra_kwargs,
            },
        }

    ## Systematic Error 
    if k == "cluster_stakes":
        if ctx.cluster_label_model is None:
            raise ValueError("cluster_label_dist sampler requires cluster_label_model in SamplerContext.")
        centroids, label_distributions = ctx.cluster_label_model
        return {
            "fn": ir_methods.sample_cluster_dist,
            "args": {
                "n_samples": ctx.n_samples,
                "centroids": centroids,
                "cluster_label_dists": label_distributions,
                **extra_kwargs,
            },
        }


    raise ValueError(f"Unknown sampler kind='{kind}'")