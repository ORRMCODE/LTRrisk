
import jax
import jax.numpy as jnp
import jax.random as jrng

jit_bincount = jax.jit(jnp.bincount, static_argnames=['length'])


@jax.jit
def scores_fn(state, model_variables, feats, rng_key):
    rng_key, rng_dropout = jrng.split(rng_key,2)
    return state.apply_fn(model_variables, feats, training=False, rngs={"dropout": rng_dropout})

@jax.jit
def scores_state_fn(state, model_variables, feats, rng_key):
    rng_key, rng_dropout = jrng.split(rng_key,2)
    logits, new_model_state = state.apply_fn(model_variables, feats, training=True, rngs={"dropout": rng_dropout}, mutable=["batch_stats"])
    return logits, new_model_state


def prd_control_fn(state, params, feats, rng_key):

    if getattr(state, "batch_stats", None) is not None:
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }
    else:
        variables = {
            "params": params}

    return scores_fn(state, variables, feats, rng_key)



def trn_control_fn(state, params, feats, rng_key):

    if getattr(state, "batch_stats", None) is not None:
        variables = {
            "params": params,
            "batch_stats": state.batch_stats,
        }
    else:
        variables = {"params": params}

    return scores_state_fn(state, variables, feats, rng_key)
    
def assign_rng_key(rng_key, no_key_dict):
    for fn_name in no_key_dict:
        rng_key, rng_x = jrng.split(rng_key, 2)
        no_key_dict[fn_name]['args'].update({'rng_key': rng_x})
    return rng_key