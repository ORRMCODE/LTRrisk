from ir_methods.control_sampling import gumbel_rankings
from ir_methods.context_sampling import sample_cluster_label_dist, sample_cluster_dist

from ir_methods.metrics import dcg_metric, ndcg_metric, KLfair_metric

from ir_methods.metrics import topK_alpha_diverse_doc
from ir_methods.metrics import N_docs_exposure, tsallis_exposure, entropy_exposure
from ir_methods.metrics import exposure_dcg_metric_fn

from ir_methods.losses import exposure_based_loss, approx_dcg_loss_fn, exposure_dcg_loss_fn
from ir_methods.losses import exposure_entropy

__all__=[
    "gumbel_rankings",
    "sample_cluster_label_dist",
    "sample_cluster_dist",
    "dcg_metric",
    "ndcg_metric",
    "KLfair_metric",
    "topK_alpha_diverse_doc",
    "exposure_based_loss",
    "approx_dcg_loss_fn",
    "exposure_dcg_loss_fn",
    "exposure_dcg_metric_fn",
    "exposure_entropy",
    "N_docs_exposure",
    "tsallis_exposure",
    "entropy_exposure",
]