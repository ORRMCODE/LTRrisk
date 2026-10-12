import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from jax.lax import stop_gradient as stop_grd
from jax.scipy import special as jspecial

import optax
from flax import linen as nn
from functools import partial

import portfolio.sto_fnctnl as fwd_fnls
import portfolio.grd_fnctnl as bwd_fnls

#####################################################################################################
def per_query_risk(risk_fn):
    def wrapped(metric_samples, aux_params, **kwargs):
        return jax.vmap(lambda m, t: risk_fn(m, t, **kwargs))(metric_samples, aux_params)
    return wrapped

def make_fn_variational(risk_fn):
    def wrapped(metric_samples, t_aux, *args, **kwargs):
        return risk_fn(metric_samples, *args, **kwargs)
    return wrapped
    
#####################################################################################################

@jax.jit
def q_EntrQ(samples, z_aux, *, alpha):
    # z_aux:   [B,]   or [B, 1]
    # samples: [B, K] or [K] (will broadcast)
    # axis=-1 logsumexp over samples
    # z_opt = 1e-4+jnn.softplus(z_aux)
    z_opt = jnn.softplus(z_aux)
    lse = jspecial.logsumexp(z_opt*samples, axis=-1)
    return (lse - jnp.log(samples.shape[-1]) - jnp.log(1.0 - alpha)) / z_opt

@partial(jax.jit, static_argnames=['signed'])
def q_CondQ(samples, t_aux, *, alpha, k=100.0, signed=True):
    # t_aux:   [B,]   or [B, 1]
    # samples: [B, K] or [K]
    # Loss is always negative we can force coherence to accelerate learning.
    if signed:
        s_sign = stop_grd(jnp.sign(jnp.mean(samples))) 
        t_opt = s_sign*jnn.softplus(t_aux)
    else:
        t_opt = t_aux
    smooth_indicator = jax.nn.sigmoid(k * (samples - t_opt))

    return t_opt + (1.0 / (1.0 - alpha)) * jnp.mean(smooth_indicator * (samples - t_opt), axis=-1)


@partial(jax.jit, static_argnames=['signed'])
def q_CondQ_H(samples, t_aux, *, alpha, signed=True):
    # t_aux:   [B,]   or [B, 1]
    # samples: [B, K] or [K]
    # t_opt = acc_constant*t_aux
    if signed:
        s_sign = stop_grd(jnp.sign(jnp.mean(samples))) 
        t_opt = s_sign*jnn.softplus(t_aux)
    else:
        t_opt = t_aux
    hard_indicator = (samples>=t_opt)
    return t_opt + (1.0 / (1.0 - alpha)) * jnp.mean(hard_indicator * (samples - t_opt), axis=-1)

#####################################################################################################
@jax.jit
def q_E(samples, t_aux):
    return fwd_fnls.expectation(samples)
@jax.jit
def upper_semideviation(samples, t_aux):
    return fwd_fnls.upper_semideviation(samples)
@jax.jit
def upper_semivariance(samples, t_aux):
    return fwd_fnls.upper_semivariance(samples)

#####################################################################################################
variational_CQ=per_query_risk(q_CondQ)
variational_EQ=per_query_risk(q_EntrQ)
expectation = per_query_risk(q_E)

u_semivar = per_query_risk(upper_semivariance)
u_semidev = per_query_risk(upper_semideviation)

variance = make_fn_variational(bwd_fnls.variance)
variance = per_query_risk(variance)

stddev = make_fn_variational(fwd_fnls.standard_deviation)
stddev = per_query_risk(stddev)
#####################################################################################################

cvar_pop = make_fn_variational(fwd_fnls.CVaR_pop)
cvar_pop = per_query_risk(cvar_pop)

cvar_wi = make_fn_variational(fwd_fnls.CVaR_Wi)
cvar_wi = per_query_risk(cvar_wi)

avg_tail = make_fn_variational(fwd_fnls.cvar_tail_average)
avg_tail = per_query_risk(avg_tail)

evar_z = make_fn_variational(fwd_fnls.Entropic_Q)
evar_z = per_query_risk(evar_z)

quantile_wi = make_fn_variational(fwd_fnls.Q_weight)
quantile_wi = per_query_risk(quantile_wi)

quantile_pop = make_fn_variational(fwd_fnls.quantile_pop)
quantile_pop = per_query_risk(quantile_pop)

#####################################################################################################