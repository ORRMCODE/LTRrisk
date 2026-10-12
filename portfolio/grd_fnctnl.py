import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import jax.nn as jnn
import jaxopt as jopt

import portfolio.sto_fnctnl as fwd_fnls

from functools import partial
from jax.lax import stop_gradient as stop_grd


@jax.jit
def variance(samples):
    n_samples=samples.size
    centered = samples - jnp.mean(samples)
    scale_grad = stop_grd(centered)*samples
    fwd = jnp.var(samples, ddof=1)
    bwd = 2/(n_samples-1)*jnp.mean(scale_grad)
    return stop_grd(fwd)+bwd-stop_grd(bwd)

### The semi-variance defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def lower_semivariance(samples):
    n_samples = samples.size

    samples_A = samples[:n_samples//2]
    samples_B = samples[n_samples//2 :]

    fwd = fwd_fnls.lower_semivariance(samples)

    indicatorA = (samples_A <= jnp.mean(samples_B))
    bwdA = jnp.mean(indicatorA * (samples_A - jnp.mean(samples_B))**2)

    indicatorB = (samples_B <= jnp.mean(samples_A))
    bwdB = jnp.mean(indicatorB * (samples_B - jnp.mean(samples_A))**2)

    bwd = 0.5 * (bwdA + bwdB)
    return stop_grd(fwd)+bwd-stop_grd(bwd)

### The semi-variance defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def upper_semivariance(samples):
    n_samples = samples.size

    samples_A = samples[:n_samples//2]
    samples_B = samples[n_samples//2 :]

    fwd = fwd_fnls.upper_semivariance(samples)
    
    indicatorA = (samples_A >= jnp.mean(samples_B))
    bwdA = jnp.mean(indicatorA * (samples_A - jnp.mean(samples_B))**2)

    indicatorB = (samples_B >= jnp.mean(samples_A))
    bwdB = jnp.mean(indicatorB * (samples_B - jnp.mean(samples_A))**2)

    bwd = 0.5 * (bwdA + bwdB)
    return stop_grd(fwd)+bwd-stop_grd(bwd)
