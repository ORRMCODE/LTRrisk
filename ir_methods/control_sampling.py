import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.scipy as jsp
import jax.scipy.special as jspecial
import jax.scipy.stats as jstats

import ir_methods.auxiliary as ir_aux

from functools import partial

def gumbel_rankings(scores, mask, rng_key, n_samples):

    _, rng_gumbel = jrng.split(rng_key, 2)
    n_docs = scores.shape[0]
    gumbel_noise = -jnp.log(-jnp.log(jrng.uniform(rng_gumbel, shape=(n_samples, n_docs))))

    masked_scores = jnp.where(
        mask[None, :],
        scores[None, :] + gumbel_noise,
        -jnp.inf
    )

    sampled_rankings = jnp.argsort(masked_scores, axis=-1)[:, ::-1]
    return sampled_rankings


