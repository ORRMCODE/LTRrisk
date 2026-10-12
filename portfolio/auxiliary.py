import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.special as jspecial

##############################################################################################
### Quantile weights for popular HF7
def W_quantile_pop(N_samples, q):
    M_alpha=(q)*(N_samples-1)+1
    w0=jnp.zeros(N_samples)
    w0=w0.at[int(jnp.floor(M_alpha))-1].set(1)
    w1=jnp.zeros(N_samples)
    w1=w1.at[int(jnp.floor(M_alpha))].set(1)
    wi=w0+(M_alpha-jnp.floor(M_alpha))*(w1-w0)
    return wi

### Quantile weights for Harrell Davis
def W_quantile_HD(N_samples, q):
    N_vec=jnp.arange(N_samples)
    w0=jspecial.betainc((N_samples+1)*(q), (N_samples+1)*(1-q), N_vec/N_samples)
    w1=jspecial.betainc((N_samples+1)*(q), (N_samples+1)*(1-q), (N_vec+1)/N_samples)
    wi=w1-w0
    return wi
##############################################################################################
