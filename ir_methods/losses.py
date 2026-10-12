import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrng

from ir_methods.exposure import exposure_prim as exposure_grad

import ir_methods.auxiliary as aux_ir
import ir_methods.control_sampling as aux_sample

from functools import partial

_TOL_LOG = 1e-16

@jax.jit
def pairwise_loss_fn(scores, labels, mask, rng_key):
    padding_mask = jnp.logical_and(mask[:,None], mask[None,:])
    greater_mask = jnp.greater(labels[None,:], labels[:,None])
    combined_mask = jnp.logical_and(padding_mask, greater_mask)
    sum_values = jnp.sum(jnp.log(1 + jnp.exp(scores[:,None]-scores[None,:])),
                        where=combined_mask)
    return sum_values/jnp.maximum(1., jnp.sum(combined_mask))

@jax.jit
def listwise_loss_fn(scores, labels, mask, rng_key):
    mask_penalty = jnp.logical_not(mask)*jnp.inf
    best_ranking = jnp.argsort(labels-mask_penalty, descending=True)
    ranked_scores = jnp.exp(scores[best_ranking]-mask_penalty)
    denominator = jnp.cumsum(ranked_scores[::-1])[::-1]
    log_prob = jnp.log(ranked_scores+1e-6) - jnp.log(denominator+1e-6)
    return -jnp.sum(log_prob, where=mask)

@jax.jit
def approx_dcg_loss_fn(scores, labels, mask, rng_key):
    pairwise_diff = scores[:,None] - scores[None,:]
    pairwise_prob = jnn.sigmoid(-pairwise_diff)
    approx_ranks = 0.5 + jnp.sum(pairwise_prob, axis=1, where=mask)
    return -jnp.sum(labels/jnp.log2(1.+approx_ranks), where=mask) 

############ Maps linear labels 0-N to values between 0 and 1 #############
@jax.jit
def exposure_based_loss(scores, labels, mask, rankings_samples, weights):

    exposures = exposure_grad(scores, weights, rankings_samples, mask=mask)
    relevances = aux_ir.rel_qd_fn(labels, 0.1)
    # relevances is a [D] vector of relevance labels per doc.
    relevance_loss = -jnp.sum(exposures * relevances, where=mask)
    return relevance_loss


@partial(jax.jit, static_argnames=["n_samples", "top_K"])
def exposure_dcg_loss_fn(scores, labels, mask, rng_key, *, n_samples, top_K):

    _, rng_ranks = jrng.split(rng_key, 2)    
    
    K = scores.shape[0] if top_K is None else top_K
    weights = jax.lax.stop_gradient(aux_ir.dcg_weights(K))

    ranking_samples = jax.lax.stop_gradient(
                            aux_sample.gumbel_rankings(scores, mask, rng_ranks, n_samples)
                        )
    exposures = exposure_grad(scores, weights, ranking_samples, mask=mask)
    relevance_loss = -jnp.sum(exposures * labels, where=mask)
    return relevance_loss


@partial(jax.jit, static_argnames=["n_samples", "top_K"])
def exposure_entropy(scores, labels, mask, rng_key, *, n_samples, top_K):
    
    _, rng_ranks = jrng.split(rng_key, 2)
    
    K = scores.shape[0] if top_K is None else top_K
    weights = jax.lax.stop_gradient(aux_ir.dcg_weights(K))

    ranks_mat = jax.lax.stop_gradient(
                        aux_sample.gumbel_rankings(scores, mask, rng_ranks, n_samples)
                )
    exposure_coeffs = exposure_grad(scores, weights, ranks_mat, mask)

    # Optional: zero out invalid docs (not strictly necessary if you always use `where=mask`)
    exposure_coeffs = jnp.where(mask, exposure_coeffs, 0.0)
    # Normalize only over valid docs
    Z = jnp.sum(exposure_coeffs, where=mask)
    Z = jnp.where(jnp.isclose(Z, 0.0, atol=1e-7), 1.0, Z)

    norm_coeffs = exposure_coeffs / Z
    # Valid docs with nonzero probability
    mask_pos = mask & ~jnp.isclose(norm_coeffs, 0.0, atol=1e-16)
    # Safe log: log(1) = 0 for masked-out entries
    safe_norm = jnp.where(mask_pos, norm_coeffs, 1.0)

    entropy = -jnp.sum(
                    safe_norm * jnp.log(safe_norm),
                    where=mask_pos,
                )
    return entropy

