import jax
import jax.numpy as jnp
from jax import jit, grad

@jit
def f(a, b, t):
    return jnp.exp(a * t + b) / (1 + jnp.exp(a * t + b))

@jit 
def quasibinomial_log_likelihood(Y, mu_hat, n_trials, overdisp):
    """
    Computes the binomial quasi log-likelihood.

    Args:
        Y (array): Observed counts of successes.
        mu_hat (array): Predicted probabilities.
        n_trials (array): Number of trials for each observation.
        overdisp (float): overdispersion estimate

    Returns:
        float: Binomial log-likelihood.
    """

    log_likelihood = overdisp * (Y * jnp.log(mu_hat) + (n_trials - Y) * jnp.log(1 - mu_hat))
    return jnp.sum(log_likelihood)


def quasibinomial_overdispersion(Y, mu_hat, n, p):
    """
    Computes the overdispersion parameter for a quasibinomial likelihood.

    Args:
        Y (array): Observed proportions (Y_i / total_trials_i).
        mu_hat (array): Predicted probabilities.
        n (int): Number of observations.
        p (int): Number of parameters in the model.

    Returns:
        float: Overdispersion estimate.
    """
    V_mu = mu_hat * (1 - mu_hat) * n   # Variance function for quasibinomial
    residuals = (Y - mu_hat * n) ** 2 / V_mu
    return jnp.sum(residuals) / (n.shape[0] - p)


def quasibinomial_residuals(Y, mu_hat, n):
    """
    Computes the Pearson residuals for a quasibinomial likelihood.

    Args:
        Y (array): Observed proportions (Y_i / total_trials_i).
        mu_hat (array): Predicted probabilities.
        n (int): Number of observations.

    Returns:
        float (array): Pearson residuals
    """
    V_mu = mu_hat * (1 - mu_hat) * n   # Variance function for quasibinomial
    residuals = (Y - mu_hat * n) ** 2 / V_mu
    return residuals


@jit
def fisher_information_matrix(a, b, xx, Y, n_trials):
    """
    Computes the Fisher Information Matrix (Σ) as the expected Hessian of the negative log-likelihood.
    """
    def neg_log_likelihood(params):
        a, b = params
        mu = f(a, b, xx)
        return -quasibinomial_log_likelihood(Y, mu, n_trials, 1.0)

    hessian_fn = jax.hessian(neg_log_likelihood)
    fisher_matrix = hessian_fn(jnp.array([a, b]))

    # Ensure it's a valid positive-definite matrix
    fisher_matrix += 1e-6 * jnp.eye(2)

    return fisher_matrix


@jit
def log_likelihood_grad(a, b, xx, Y, n_trials):
    """ Compute the gradient of the log-likelihood. """
    def log_likelihood(params):
        a, b = params
        mu = f(a, b, xx)
        return quasibinomial_log_likelihood(Y, mu, n_trials, 1.0)

    return grad(log_likelihood)(jnp.array([a, b]))  # JIT-compiled gradient