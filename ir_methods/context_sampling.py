
import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.scipy as jsp
import jax.nn as jnn
import jax.scipy.special as jspecial
import jax.scipy.stats as jstats
import optax

import ir_methods.auxiliary as ir_aux

from functools import partial

from ir_methods.context.LabelClusterDistribution import assign_clusters_with_label_distributions as assign_label_distribution

########################## Sampling Labels from Uncertainty model #################################
@partial(jax.jit, static_argnames=['n_samples'])
def sample_cluster_label_dist(feats, labels, mask, rng_key, n_samples, *, centroids, cluster_label_dists):

    _, rng_fn, rng_samples = jrng.split(rng_key, 3)
    _, label_dist_qd = assign_label_distribution(feats, labels, mask, centroids, cluster_label_dists)

    logits = jnp.clip(label_dist_qd, 1e-9, 1.0-1e-9)
    logits = jnp.log( logits )

    samples_SQD = jax.random.categorical(
                        rng_samples,
                        logits=logits,
                        axis=-1,        # classes along last dimension C
                        shape=(n_samples, *labels.shape)
                )
    samples_SQD = ir_aux.pow_label(samples_SQD)

    return jnp.transpose(samples_SQD, (1, 2, 0))

#####################################################################################################
@partial(jax.jit, static_argnames=['n_samples'])
def sample_cluster_dist(feats, labels, mask, rng_key, n_samples, *, centroids, cluster_label_dists):
    """
    feats:      (Q, D, F)
    labels:     (Q, D)
    mask:       (Q, D)
    centroids:  (K, F)
    cluster_label_dists: (K, L)  # per-cluster distribution over L labels
    returns:
        samples_QDS: (Q, D, S)
    """
    Q_queries, D_docs, _ = feats.shape
    K_clusters, L_labels = cluster_label_dists.shape

    _ , rng_samples = jrng.split(rng_key, 2)

    # 1) Get cluster IDs per (q, d): shape (Q, D)
    cluster_ids, _ = assign_label_distribution(
        feats, labels, mask, centroids, cluster_label_dists
    )   # cluster_ids: (Q, D), values in [0, K-1]

    # 2) Compute logits per cluster, then broadcast over Q
    cluster_prob_dist = jnp.array(cluster_label_dists)      # (K, L)
    logits_KL = jnp.clip(cluster_prob_dist, 1e-9, 1.0 - 1e-9)
    logits_KL = jnp.log(logits_KL)                          # (K, L)

    # Broadcast to per-(q, k) logits: (Q, K, L)
    logits_QKL = jnp.broadcast_to(logits_KL[None, :, :], (Q_queries, K_clusters, L_labels))

    # 3) Sample S times for every (q, k)
    # logits_QKL: batch shape (Q, K), class axis = -1
    # shape=(n_samples,) => output shape (S, Q, K)
    samples_SQK = jax.random.categorical(
        rng_samples,
        logits=logits_QKL,
        axis=-1,          # classes along last dim L
        shape=(n_samples, Q_queries, K_clusters)
    )                       # (S, Q, K)

    # Reorder to (Q, K, S)
    samples_QKS = jnp.transpose(samples_SQK, (1, 2, 0))    # (Q, K, S)

    # 4) Map each (q, d) to its cluster k, using query index q and cluster_ids[q, d]
    q_idx = jnp.arange(Q_queries)[:, None]                         # (Q, 1)
    q_idx_bd = jnp.broadcast_to(q_idx, cluster_ids.shape)  # (Q, D)

    # For each (q, d), pick [q, cluster_ids[q, d], :]
    samples_QDS = samples_QKS[q_idx_bd, cluster_ids, :]    # (Q, D, S)

    #Label transform
    samples_QDS = ir_aux.pow_label(samples_QDS)

    return samples_QDS
