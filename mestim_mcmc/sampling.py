import jax
import jax.numpy as jnp
from jax import jit, grad
from tqdm import tqdm
from .scores import quasibinomial_log_likelihood, quasibinomial_overdispersion, quasibinomial_residuals, fisher_information_matrix, log_likelihood_grad, f

def step(a, b, s, key, xx, Y, n_trials, step_size, fisher_inv, sqrt_fisher_inv, quasi):
    """
    Performs a **single Langevin update step** for (a, b, s).

    Args:
        a, b, s: Current parameter values.
        key: Random key for JAX.
        xx, Y, n_trials: Data.
        step_size: Langevin step size.
        fisher_inv: Inverse Fisher Information Matrix (precomputed).
        sqrt_fisher_inv: Cholesky decomposition of fisher_inv.
        quasi: Quasi-likelihood model ("MoM" or "random").

    Returns:
        Updated (a, b, s), new mu, and updated key.
    """
    key, noise_key1, noise_key2, noise_key3 = jax.random.split(key, 4)

    # Compute gradient of log-likelihood
    grad_vec = log_likelihood_grad(a, b, xx, Y, n_trials)

    # Preconditioned Gradient Update
    preconditioned_grad = fisher_inv @ grad_vec  

    # Langevin Noise
    noise_vec = jax.random.normal(noise_key1, (2,))
    noise_term = sqrt_fisher_inv @ noise_vec

    # Langevin Update for (a, b)
    update = (step_size / 2) * preconditioned_grad + jnp.sqrt(step_size) * noise_term * jnp.sqrt(s)
    a, b = jnp.array([a, b]) + update  # Update parameters

    # Compute mu
    mu = f(a, b, xx)

    # **Update Overdispersion s (if quasi="random")**
    if quasi == "random":
        pearson_residuals = quasibinomial_residuals(Y, mu, n_trials)
        sum_pearson_residuals = jnp.sum(pearson_residuals)
        df = xx.shape[0] - 2

        # Fisher Information for s
        fisher_s = df / (2 * s**2)

        # Gradient of Gamma log-likelihood for s
        grad_s = -((df / (2 * s)) - (sum_pearson_residuals / (2 * s**2)))

        # Langevin Update for s
        noise_s = jax.random.normal(noise_key3) * jnp.sqrt(1 / fisher_s)
        s += (step_size / 2) * grad_s / fisher_s + jnp.sqrt(step_size) * noise_s

        # Ensure positivity of s
        s = jnp.maximum(s, 1e-6)

    elif quasi == "MoM":
        s = quasibinomial_overdispersion(Y, mu, n_trials, p=2)

    return a, b, s, mu, key
step = jit(step, static_argnames=["quasi"])  # JIT with quasi as static


def preconditioned_ULA(Y, xx, n_trials, num_samples=1000, step_size=0.1, quasi=False, fisher_updates=1, key=jax.random.PRNGKey(42)):
    """
    Preconditioned Unadjusted Langevin Algorithm (P-ULA) with Fisher Information Matrix.

    Args:
        Y (array): Observed successes.
        xx (array): Time series data.
        n_trials (array): Number of trials per observation.
        num_samples (int): Number of MCMC iterations.
        step_size (float): Step size for Langevin updates.
        fisher_updates (int): How often to recompute the Fisher matrix.
        key (jax.random.PRNGKey): Random key for JAX.

    Returns:
        a_samples, b_samples, s_samples, mu_samples: MCMC samples of (a, b, s, mu).
    """
    # Initialize parameters
    a, b, s = 0.0, 0.0, 1.0
    a_samples, b_samples, s_samples, mu_samples = [], [], [], []

    for i in tqdm(range(num_samples)):
        if i % fisher_updates == 0:  # Update Fisher matrix every `fisher_updates` steps
            fisher_matrix = fisher_information_matrix(a, b, xx, Y, n_trials)
            fisher_inv = jnp.linalg.inv(fisher_matrix)
            sqrt_fisher_inv = jnp.linalg.cholesky(fisher_inv)

        # Perform **one step of P-ULA**
        a, b, s, mu, key = step(a, b, s, key, xx, Y, n_trials, step_size, fisher_inv, sqrt_fisher_inv, quasi)

        # Store samples
        a_samples.append(a)
        b_samples.append(b)
        s_samples.append(s)
        mu_samples.append(mu)

    return jnp.array(a_samples), jnp.array(b_samples), jnp.array(s_samples), jnp.array(mu_samples)
