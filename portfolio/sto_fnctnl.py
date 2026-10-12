import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import jax.nn as jnn
import jaxopt as jopt
from functools import partial

@jax.jit
def expectation(samples):
    return jnp.mean(samples)

@jax.jit
def variance(samples):
    return jnp.var(samples, ddof=1)

@jax.jit
def standard_deviation(samples):
    return jnp.std(samples, ddof=1)

### The semi-variance defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def lower_semivariance(samples):
    avg_samples = jnp.mean(samples)
    indicator_samples = (samples<=avg_samples)
    return(jnp.mean(((samples-avg_samples)**2)*indicator_samples))

### The semi-variance defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def upper_semivariance(samples):
    avg_samples = jnp.mean(samples)
    indicator_samples = (samples>=avg_samples)
    return(jnp.mean(((samples-avg_samples)**2)*indicator_samples))

### The semi-semideviation defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def lower_semideviation(samples):
    avg_samples = jnp.mean(samples)
    indicator_samples = (samples<=avg_samples)
    return jnp.sqrt(jnp.mean(((samples-avg_samples)**2)*indicator_samples))

### The semi-semideviation defined in a straightforward way does not normalize by the number of effective samples.
@jax.jit
def upper_semideviation(samples):
    avg_samples = jnp.mean(samples)
    indicator_samples = (samples>=avg_samples)
    return jnp.sqrt(jnp.mean(((samples-avg_samples)**2)*indicator_samples))

@jax.jit
def quantile_pop(samples, q):
    return jnp.quantile(samples, q)

### Direct Estimation of quantiles
@jax.jit
def Q_weight(samples, wi):
    sorted_loss = jnp.sort(samples)
    return(jnp.sum(wi*sorted_loss))

# No @jit here — called inside vmap/jit, alpha arrives as concrete Python float
def cvar_tail_average(losses: jnp.ndarray, alpha: float = 0.05) -> jnp.ndarray:
    losses_sorted = jnp.sort(losses)
    n = losses_sorted.shape[0]                    # static at trace time
    k = max(0, min(int((alpha) * n), n - 1))  # Python int: safe because alpha is concrete
    return losses_sorted[k:].mean()

@partial(jax.jit, static_argnames=('alpha', 'leave_out', 'n_boot'))
def bootstrap_CVaR_tail(
    losses: jnp.ndarray,
    alpha: float = 0.05,
    leave_out: float = 0.1,
    n_boot: int = 1000,
    rng_boot_key: jax.Array = jax.random.PRNGKey(0),
) -> jnp.ndarray:
    n = losses.shape[0]
    m = max(1, int((1.0 - leave_out) * n))        # Python int: safe, leave_out is static

    keys = jax.random.split(rng_boot_key, n_boot)

    def one_bootstrap(k):
        idx = jax.random.choice(k, n, shape=(m,), replace=False)
        return cvar_tail_average(losses[idx], alpha=alpha)  # alpha is concrete here

    return jax.vmap(one_bootstrap)(keys).mean()


@jax.jit
def entropic_fun(z_aux, samples, alpha):
    lse = jspecial.logsumexp(jnn.softplus(z_aux)*samples)
    return (lse - jnp.log(samples.shape[0])-jnp.log(1-alpha)) / jnn.softplus(z_aux)


@partial(jax.jit, static_argnames=['tol', 'maxiter', 'linesearch'])
def Entropic_Q(samples, *, alpha=0.95, tol=1e-6, maxiter=75, linesearch="zoom"):

    # Objective for LBFGS: explicit args (z_aux, samples, alpha)
    def ent_z(z_aux, samples_, alpha_):
        # No stop_gradient here; LBFGS + implicit_diff needs full dependence
        return entropic_fun(z_aux, samples_, alpha_)

    min_z = jopt.LBFGS(
        fun=ent_z,
        tol=tol,
        maxiter=maxiter,
        linesearch=linesearch,
        implicit_diff=True
    )

    # Pass samples and alpha explicitly to run()
    sol_A = min_z.run(0.01, samples, alpha)
    sol_B = min_z.run(0.1, samples, alpha)
    sol_C = min_z.run(1.0, samples, alpha)

    zA_star = sol_A.params
    zB_star = sol_B.params
    zC_star = sol_C.params

    # Stop gradients w.r.t. the optimizer *output* z*, so gradients
    # only flow via implicit differentiation of the argmin.
    opt_qalpha_A = entropic_fun(jax.lax.stop_gradient(zA_star), samples, alpha)
    opt_qalpha_B = entropic_fun(jax.lax.stop_gradient(zB_star), samples, alpha)
    opt_qalpha_C = entropic_fun(jax.lax.stop_gradient(zC_star), samples, alpha)

    return jnp.nanmin(jnp.array([opt_qalpha_A, opt_qalpha_B, opt_qalpha_C]))

@jax.jit
def CA_Q_MC(samples, alpha, Qalpha):
    return Qalpha+(1/(1-alpha))*jnn.relu(samples-Qalpha).mean()

@jax.jit
def CVaR_Wi(samples, *, alpha, wi):
    q_optimized_alpha = Q_weight(samples, wi)
    return CA_Q_MC(samples, alpha, q_optimized_alpha)

@jax.jit
def CVaR_pop(samples, *, alpha):
    q_optimized_alpha = jnp.quantile(samples, alpha)
    return CA_Q_MC(samples, alpha, q_optimized_alpha)
