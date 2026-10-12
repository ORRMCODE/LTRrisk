import jax
import jax.numpy as jnp
import jax.random as jrng
from functools import wraps

def accept_rng_key(fn):
    @wraps(fn)
    def wrapper(*args, rng_key=None, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

def per_query_metric(fn):
    def wrapped(preds, targets, mask, **kwargs):
        return jax.vmap(lambda p,t,m: fn(p,t,m, **kwargs))(preds,targets,mask)
    return wrapped

def per_query_risk(fn):    
    def wrapped(samples, **kwargs):
        return jax.vmap(lambda x: fn(x, **kwargs))(samples)    
    return wrapped

def per_query_loss(fn):
    def wrapped(scores, labels, mask, **kwargs):
        return jax.vmap(lambda s, l, m: fn(p,t,m, **kwargs))(scores, labels, mask)
    return wrapped

def per_query_metric_rng(metric_fn):
    def wrapped(preds, targets, mask, rng_key, **kwargs):
        keys = jrng.split(rng_key, preds.shape[0])
        return jax.vmap(lambda p, t, m, k: metric_fn(p, t, m, rng_key=k, **kwargs))(
            preds, targets, mask, keys
        )
    return wrapped

def per_query_risk_rng(risk_fn):
    def wrapped(metric_samples, rng_key, **kwargs):
        keys = jrng.split(rng_key, metric_samples.shape[0])
        return jax.vmap(lambda x, k: risk_fn(x, rng_key=k, **kwargs))(metric_samples, keys)
    return wrapped

def per_query_loss_rng(loss_fn):
    def wrapped(scores, labels, mask, rng_key, **kwargs):
        keys = jrng.split(rng_key, scores.shape[0])
        return jax.vmap(lambda s, l, m, k: loss_fn(s, l, m, rng_key=k, **kwargs))(
            scores, labels, mask, keys
        )
    return wrapped


def per_sample_metric(metric_fn):
    def wrapped(preds, targets, mask, rng_key, **kwargs):
        return jax.vmap(
                    lambda label_sample: metric_fn(preds, label_sample, mask, rng_key=rng_key, **kwargs),
                    in_axes = -1,
                    out_axes = -1
                )(targets)
    return wrapped

def per_sample_loss(loss_fn):
    def wrapped(scores, labels, mask, rng_key, **kwargs):
        return jax.vmap(
                    lambda label_sample: loss_fn(scores, label_sample, mask, rng_key=rng_key, **kwargs),
                    in_axes = -1,
                    out_axes = -1
                )(labels)
    return wrapped



def polarity_transform(func):
    def wrapper(samples, *args, **kwargs):
        return -func(-samples, *args, **kwargs)
    return wrapper

def reflection_transform(func):
    def wrapper(samples, *args, **kwargs):
        return func(-samples, *args, **kwargs)
    return wrapper

def negation_transform(func):
    def wrapper(samples, *args, **kwargs):
        return -func(samples, *args, **kwargs)
    return wrapper

# -------------------- Auxiliary functions -------------------- #
def query_level_fnal(metric_fn, risk_fn, *, risk_args):
    """
    Local = average of risk
    """
    def fn(preds, sampled_targets, mask, rng_key, **kwargs):
        """
        preds: (Q,D)
        sampled_targets: (Q,D,S)
        mask: (Q,D)
        rng_key: JAX PRNG
        """
        # metric_samples: (Q,S)
        metric_samples = metric_fn(preds, sampled_targets, mask, rng_key, **kwargs)
        # jax.debug.print("metric_values_avg: {}, {}", metric_samples.mean(axis=-1), metric_samples.shape)
        # risk per query over S
        per_query_risk = jax.vmap( lambda qs: risk_fn(qs, **risk_args), in_axes=0)(metric_samples)
        # jax.debug.print("risk_values_avg{}", per_query_risk)
        return per_query_risk

    return jax.jit(fn)

# -------------------- Auxiliary functions -------------------- #
def batch_level_fnal(metric_fn, risk_fn, *, risk_args):
    """
    Local = average of risk
    """
    def fn(preds, sampled_targets, mask, rng_key, **kwargs):
        """
        Global = risk of average
        preds: (Q,D)
        sampled_targets: (Q,D,S)
        mask: (Q,D)
        """
        # mean over queries
        metric_samples = metric_fn(preds, sampled_targets, mask, rng_key, **kwargs)
        per_sample_mean = jnp.mean(metric_samples, axis=0)
        return risk_fn(per_sample_mean, **risk_args)
        
    return jax.jit(fn)
# -------------------- Auxiliary functions -------------------- #
