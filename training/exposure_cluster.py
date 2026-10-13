import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn
import numpy as np

import ir_methods.exposure as aux_exp
import ir_methods.auxiliary as aux_ir
from ir_methods.control_sampling import gumbel_rankings

from functools import partial

def per_query_metric_rng(metric_fn):
    @partial(jax.jit, static_argnames=('n_rank_samples', 'top_K'))
    def wrapped(scores, feats, labels, mask, rng_key, **kwargs):
        keys = jrng.split(rng_key, scores.shape[0])
        return jax.vmap(lambda s, f, l, m, k: metric_fn(s, f, l, m, rng_key=k, **kwargs))(
            scores, feats, labels, mask, keys
        )
    return wrapped


@jax.jit
def _assign_clusters(x, centroids):
    """
    Assign each point in x to the nearest centroid (squared Euclidean).

    x:         [N, F']
    centroids: [K, F']
    returns:   cluster indices [N] (int)
    """
    distances = jnp.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(distances, axis=1)

@partial(jax.jit, static_argnames=['n_rank_samples','top_K'])
def exposure_per_cluster(scores, feats, labels, mask, *, rng_key, n_rank_samples, top_K, centroids):

    _, rng_ranks = jrng.split(rng_key, 2)
    weights = aux_ir.dcg_weights(top_K)
   
    n_clusters, n_feats = centroids.shape
    cluster_exposure = jnp.zeros(n_clusters)
    cluster_n_docs = jnp.zeros(n_clusters)
    
    clusters_d = _assign_clusters(feats, centroids)
    
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_rank_samples)
    exposure_d = aux_exp.exposure(scores, weights, ranks_mat, mask)
    # Optional: zero out invalid docs (not strictly necessary if you always use `where=mask`)
    exposure_d = jnp.where(mask, exposure_d, 0.0)

    cluster_exposure=cluster_exposure.at[clusters_d].add(exposure_d)
    cluster_n_docs = cluster_n_docs.at[clusters_d].add(mask)
    return cluster_exposure, cluster_n_docs

v_exposure_per_cluster = per_query_metric_rng(exposure_per_cluster)



def dataset_exposure_per_cluster(dataset, trainer, model_state, rng_key, n_rank_samples, top_K, cluster_centroids):

    cluster_exposure = jnp.zeros(cluster_centroids.shape[0])
    cluster_n_docs = jnp.zeros(cluster_centroids.shape[0])       

    for batch_data in dataset:
        _, rng_loss = jrng.split(rng_key)
        batch_feats, batch_labels, batch_mask, batch_qids = batch_data
        scores_q_d, _ = trainer.trn_control_fn(
                            model_state, 
                            model_state.params, 
                            batch_feats, 
                            rng_loss
                    )
        cluster_exposure_q, cluster_n_docs_q = v_exposure_per_cluster(
                                                    scores_q_d,
                                                    batch_feats,
                                                    batch_labels, 
                                                    batch_mask,
                                                    rng_loss,
                                                    n_rank_samples = n_rank_samples,
                                                    top_K = int(top_K),
                                                    centroids = cluster_centroids
                                            )
        cluster_exposure += jnp.sum(cluster_exposure_q, axis=0)
        cluster_n_docs += jnp.sum(cluster_n_docs_q, axis=0)

    return cluster_exposure, cluster_n_docs