import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.lax as jlax


import auxiliary.wrappers as aux_wrap
import ir_methods as ir_methods

from auxiliary import trn_control_fn, prd_control_fn
from functools import partial

from ir_methods.auxiliary import dcg_weights
from ir_methods.control_sampling import gumbel_rankings
from ir_methods.exposure import exposure_prim


@partial(jax.jit, static_argnames=["n_samples", "top_K"])
@aux_wrap.per_query_loss_rng
def v_exposure_loss_test(scores, labels, mask, rng_key, *, n_samples, top_K):
    return ir_methods.exposure_dcg_loss_fn(scores, labels, mask, rng_key, n_samples=n_samples, top_K=top_K)

@aux_wrap.per_query_loss_rng
def v_exposure_loss_test_grad(scores, labels, mask, rng_key, *, n_samples, top_K):
    vexpo_loss = lambda x: ir_methods.exposure_dcg_loss_fn(x, labels, mask, rng_key, n_samples=n_samples, top_K=top_K)
    v_grad = jax.grad(vexpo_loss)
    return v_grad(scores)

@partial(jax.jit, static_argnames=["n_samples", "top_K"])
@aux_wrap.per_query_loss_rng
def v_exposure_prim(scores, labels, mask, rng_key, *, n_samples, top_K):
    
    _, rng_ranks = jrng.split(rng_key, 2)    
    
    K = scores.shape[0] if top_K is None else top_K
    weights = dcg_weights(K)

    ranking_samples = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    exposures = exposure_prim(scores, weights, ranking_samples, mask=mask)

    return exposures



def verbose_0(state, batch_data, loss_fn_params, rng_loss, rng_dropout):
    feat, labels, mask = batch_data
    scores = prd_control_fn(state, state.params, feat, rng_dropout)

    expos_test = v_exposure_prim(scores, labels, mask, rng_loss, **loss_fn_params)
    eloss_test = v_exposure_loss_test(scores, labels, mask, rng_loss, **loss_fn_params)
    grads_test = v_exposure_loss_test_grad(scores, labels, mask, rng_loss, **loss_fn_params)
    
    means_eval_fn = lambda x: jnp.mean(v_exposure_loss_test(x, labels, mask, rng_loss, **loss_fn_params))

    grads_mean_test = jax.grad(means_eval_fn)(scores)

    print(f'scores: {scores.shape}', jnp.min(scores), jnp.mean(scores), jnp.max(scores))
    print(f'eloss_test: {eloss_test.shape}', jnp.min(eloss_test), jnp.mean(eloss_test), jnp.max(eloss_test))
    print(f'expos_test: {expos_test.shape}', jnp.min(expos_test), jnp.mean(expos_test), jnp.max(expos_test))
    print(f'grads_test: {grads_test.shape}', jnp.min(grads_test), jnp.mean(grads_test), jnp.max(grads_test))
    print(f'grads_test_mean: {grads_mean_test.shape}', jnp.min(grads_mean_test), jnp.mean(grads_mean_test), jnp.max(grads_mean_test))


def verbose_1(state):
    dic_batch_norm = state["batch_stats"]
    for layer_name, dic_values in dic_batch_norm.items():
        mean_values = dic_values["mean"]
        var_values = dic_values["var"]
        print(f'{layer_name} MEAN:', jnp.min(mean_values), jnp.mean(mean_values), jnp.max(mean_values))
        print(f'{layer_name} VAR:', jnp.min(var_values), jnp.mean(var_values), jnp.max(var_values))