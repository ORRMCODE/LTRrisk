
import jax
import jax.numpy as jnp
import jax.random as jrng

HIGHEST_LABEL=4
MIN_MAX_REL=0.9
MAX_MIN_REL=0.1

###########################################################################
@jax.jit
def rating_label(label):
    return(jnp.log2(label+1))

@jax.jit
def pow_label(label):
    return 2**label-1


@jax.jit
def rel_qd_fn(labels_qd, epsilon_min=MAX_MIN_REL):
    rel_qd = (1./(HIGHEST_LABEL))*MIN_MAX_REL*rating_label(labels_qd)
    rel_qd += epsilon_min
    return rel_qd

###########################################################################
@jax.jit
def ULIN_rel_qd_ifn(rel_qd, epsilon_min = MAX_MIN_REL):
    urate = HIGHEST_LABEL*(rel_qd-epsilon_min)/MIN_MAX_REL
    return jnp.round(urate, 0)

@jax.jit
def rel_qd_ifn(rel_qd):
    fixed_labels=pow_label(jnp.arange(HIGHEST_LABEL+1))
    fixed_relevances=rel_qd_fn(fixed_labels)

    dist_mat = jnp.abs(rel_qd[:, None] - fixed_relevances[None, :])
    indices_labels = jnp.argmin(dist_mat, axis=1)
    closest_val = fixed_relevances[indices_labels]

#    new_vals = N_U_LABELS*(closest_val-MAX_MIN_REL)/MIN_MAX_REL
    new_vals = ULIN_rel_qd_ifn(closest_val)
    return jnp.round(2**new_vals-1)

##########################################################################
def dcg_weights(K):
    return 1/jnp.log2(jnp.arange(2,2+K))