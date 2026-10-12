# training_specs.py

from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Optional

import jax
import jax.numpy as jnp

import auxiliary as aux
import ir_methods
import portfolio as portfolio
import auxiliary as aux
import auxiliary.wrappers as aux_wrap

from functools import partial

@jax.jit
def conservative_sampling(feats, labels, mask, rng_key):
    return ir_methods.homoscedastic_params(
        feats,
        labels,
        mask,
        rng_key,
        concentration_value=100,
        q_temperature=0.0,
    )

@jax.jit
def deterministic_sampling(feats, labels, mask, rng_key):
    return ir_methods.homoscedastic_params(
        feats,
        labels,
        mask,
        rng_key,
        concentration_value=1000,
        q_temperature=0.0,
    )

def trade_off_cost(samples, *, cost, gain_fnal, risk_fnal, gain_fnal_args, risk_fnal_args):
    return gain_fnal(samples, **gain_fnal_args)+cost*risk_fnal(samples, **risk_fnal_args)

def vari_trade_off_cost(samples, t_aux, *, cost, gain_fnal, risk_fnal, gain_fnal_args, risk_fnal_args):
    gain = gain_fnal(samples, t_aux, **gain_fnal_args)
    risk = risk_fnal(samples, t_aux, **risk_fnal_args)
    return gain+cost*risk