import jax
import jax.numpy as jnp
from .psifun import huber_psi


# Define function f(a, b, t)
def f(a, b, t):
    return jnp.exp(a * t + b) / (1 + jnp.exp(a * t + b))

def robust_mean(x, psi_fun=huber_psi, mu_start=None, delta=1.0, tol=1e-2, max_iter=100):
    """Computes the robust mean from an influence function, psi_fun using a reweighing scheme."""
    if mu_start is None:
        mu = jnp.median(x)  # Start with median for robustness
    else:
        mu = mu_start
    
    def cond_fn(state):
        mu, mu_prev, _ = state
        return jnp.abs(mu - mu_prev) > tol
    
    def body_fn(state):
        mu, _, iter_count = state
        residuals = x - mu
        psi_values = huber_psi(residuals, delta)
        weight_sum = jnp.sum(jnp.where(residuals != 0, psi_values / residuals, 1.0))
        mu_new = mu + jnp.sum(psi_values) / weight_sum
        return mu_new, mu, iter_count + 1
    
    mu_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (mu, mu + 2 * tol, 0))
    return mu_final


def pearson_residuals(y, fitted, variance):
    """
    Computes Pearson residuals.

    Args:
        y (array): Observed values.
        fitted (array): Fitted (predicted) values.
        variance (array): Variance function values.

    Returns:
        array: Pearson residuals.
    """
    return (y - fitted) / jnp.sqrt(variance)


def make_fit_grad_fn(x, fit_fn):
    """
    Creates a function to compute the gradient of fitted values w.r.t. parameters.

    Args:
        x (array): Predictor values.
        fit_fn (function): Function computing fitted values as fit_fn(x, theta).

    Returns:
        function: A function that computes the Jacobian of fitted values.
    """
    def fitted_fun(theta):
        return fit_fn(x, theta)
    return jax.jacfwd(fitted_fun)  # Compute Jacobian w.r.t. theta


def make_score_fun(
    x, y,
    influence_fn,   
    residual_fn=pearson_residuals, 
    fit_fn=None, 
    var_fn=None,
    fit_grad_fn=None 
):
    """
    Constructs a score function for estimating parameters via M-estimation.

    Args:
        x (array): Predictor values.
        y (array): Observed response values.
        influence_fn (function): Influence function applied to residuals.
        residual_fn (function, optional): Function computing residuals. Defaults to `pearson_residuals`.
        fit_fn (function): Function computing fitted values as fit_fn(x, theta).
        var_fn (function): Function computing variance estimates based on fitted values.
        fit_grad_fn (function, optional): Function computing the Jacobian of fitted values.

    Returns:
        function: Score function that takes `theta` as input and computes the score.
    """
    if fit_grad_fn is None:
        fit_grad_fn = make_fit_grad_fn(x, fit_fn)  # Compute ∇ fitted(x, theta)

    def score_fun(theta):
        """
        Computes the score function for given parameter values theta.

        Args:
            theta (array): Current parameter vector.

        Returns:
            array: Score function value (gradient of loss).
        """
        fitted_vals = fit_fn(x, theta)  # Compute fitted values
        variance_vals = var_fn(fitted_vals)  # Compute variance estimates
        residual_vals = residual_fn(y, fitted_vals, variance_vals)  # Compute residuals
        influence_vals = influence_fn(residual_vals)  # Compute influence values

        # Compute score: influence * fitted gradient
        score = influence_vals[:, None] * fit_grad_fn(theta)  # Broadcasting over all observations

        return jnp.sum(score, axis=0)  # Sum over all observations (TODO: averaging)

    return score_fun
