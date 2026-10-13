import jax
import jax.nn as jnn
import jax.numpy as jnp
from functools import partial

from flax import linen as nn

_DROPOUT_RATE= 0.5
_BATCHNORM_MOMENTUM = 0.999
_EPSILON_VAR = 1e-3

@partial(jax.jit, static_argnames=["tau"])
def smooth_clipper(x, tau=50):
    return tau*jnp.tanh(x/tau)

@partial(jax.jit, static_argnames=["ub", "lb"])
def Asysmooth_clipper(x, ub=600, lb=-600):
    center = (ub+lb)/2
    dist = (ub-lb)/2
    return center + dist*jnp.tanh((x-center)/dist)


class NNAuxVariational(nn.Module):
    vocab_size: int
    embed_dim: int = 1

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
        )(x)
        return x


class NNPolicy(nn.Module):

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(256)(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=_BATCHNORM_MOMENTUM, epsilon=_EPSILON_VAR)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=_DROPOUT_RATE, deterministic=not training)(x)
        x = nn.Dense(128)(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=_BATCHNORM_MOMENTUM, epsilon=_EPSILON_VAR)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=_DROPOUT_RATE, deterministic=not training)(x)
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=_BATCHNORM_MOMENTUM, epsilon=_EPSILON_VAR)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=_DROPOUT_RATE, deterministic=not training)(x)
        x = nn.Dense(1)(x)
        x = Asysmooth_clipper(x)
        return jnp.squeeze(x, axis=-1)