# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: mestim-jax
#     language: python
#     name: mestim-jax
# ---

import jax.numpy as jnp
import jax
import jax.random as random
import matplotlib.pyplot as plt
from moulax.sampling import step, preconditioned_ULA, make_fisher_matrix
import arviz as az

# # Simulate data
#
# We simulate data from a gaussian linear model. We contaminate with two outliers with high leverage.

# +
# Set seed for reproducibility
key = random.PRNGKey(42)

# Generate x values
xx = jnp.arange(20)

# Define true function parameters
true_a = 1.0
true_b = 0.0

# Compute true y values
y_true = true_a * xx + true_b

# Add Gaussian noise to create observed y values
noise_std = 1.0  # Standard deviation of noise
key, subkey = random.split(key)
y_obs = y_true + random.normal(subkey, shape=xx.shape) * noise_std

# Perturb last two data points
y_obs = y_obs.at[-2:].set(0)

# Convert to NumPy for plotting
xx_np = jnp.array(xx)
y_obs_np = jnp.array(y_obs)
y_true_np = jnp.array(y_true)

# -

# # Define the gradients
#
# We define all the functions necessary to get the gradients.

# +
def fitted(x, theta):
    """Computes fitted values y = a*x + b."""
    a, b = theta
    return a * x + b

def residual(y, y_fit):
    """Computes residuals (difference between observed and fitted values)."""
    return y - y_fit

def ols_influence(residuals):
    """Ordinary Least Squares (OLS) influence function (identity)."""
    return residuals

def mad_influence(residuals):
    """influence function for mean absolute deviation. CANNOT WORK WITH FISHER CONDIITONING""" 
    return jnp.sign(residuals)

def pseudo_huber_influence(residuals, delta=1.0):
    """
    Computes the influence function for the Pseudo-Huber loss.

    Args:
        residuals (array): Residual values.
        delta (float): Pseudo-Huber threshold (default = 1.0).

    Returns:
        array: Pseudo-Huber influence function values.
    """
    return residuals / jnp.sqrt(1 + (residuals / delta) ** 2)


def make_fitted_grad(x):
    """
    Returns a function that computes the gradient of fitted values w.r.t. theta.
    """
    def fitted_fun(theta):
        return fitted(x, theta)
    return jax.jacfwd(fitted_fun)

def make_score_fun(x, y, influence=ols_influence):
    """
    Returns a score function that takes `theta` as input.

    Args:
        x (array): Predictor values.
        y (array): Observed values.
        influence (function): Influence function applied to residuals.

    Returns:
        score_fun (function): Function that computes score based on theta.
    """
    fitted_grad_fn = make_fitted_grad(x)  # Compute ∇ fitted(x, theta)

    def score_fun(theta):
        fitted_vals = fitted(x, theta)  # Compute fitted values
        residual_vals = residual(y, fitted_vals)  # Compute residuals
        influence_vals = influence(residual_vals)  # Compute influence values

        # Compute score: influence * fitted gradient
        score = influence_vals[:, None] * fitted_grad_fn(theta)  # Broadcasting

        return jnp.sum(score, axis=0)  # Sum over all observations

    return score_fun  # JIT-compiled for efficiency



# -

# # Sample

# +
ols_grad = make_score_fun(xx, y_obs, ols_influence)
init_theta = jnp.array([0.0, 0.0])  

nonrobust_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=ols_grad
)


# +
pseudo_huber_grad = make_score_fun(xx, y_obs, pseudo_huber_influence)
init_theta = jnp.array([0.0, 0.0]) 

pseudo_huber_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=pseudo_huber_grad
)

# -

az.summary(nonrobust_samples)


az.summary(nonrobust_samples.sel(draw=slice(1000, None)))


az.summary(pseudo_huber_samples)

az.summary(pseudo_huber_samples.sel(draw=slice(1000, None)))


# # Plot 

# +
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Extract samples from ArviZ InferenceData
nonrobust_samples = nonrobust_samples.posterior["theta"].mean(dim="chain").values  # Shape (num_draws, num_params)
pseudo_huber_samples = pseudo_huber_samples.posterior["theta"].mean(dim="chain").values

# Plot non-robust samples
axes[0].plot(jnp.arange(nonrobust_samples.shape[0]), nonrobust_samples[:, 0], label="Non-robust")
axes[1].plot(jnp.arange(nonrobust_samples.shape[0]), nonrobust_samples[:, 1], label="Non-robust")

# Plot Pseudo-Huber samples
axes[0].plot(jnp.arange(pseudo_huber_samples.shape[0]), pseudo_huber_samples[:, 0], label="Pseudo-Huber")
axes[1].plot(jnp.arange(pseudo_huber_samples.shape[0]), pseudo_huber_samples[:, 1], label="Pseudo-Huber")

# Plot baseline (true values)
axes[0].axhline(true_a, linestyle="dashed", color="black", label="True slope")
axes[1].axhline(true_b, linestyle="dashed", color="black", label="True intercept")

# Titles
axes[0].set_title("Slope traceplot")
axes[1].set_title("Intercept traceplot")

# Collect unique legend handles
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))

fig.tight_layout()
plt.show()
# -

ols_fit = fitted(xx, nonrobust_samples.mean(axis=0))
pseudo_huber_fit = fitted(xx, pseudo_huber_samples.mean(axis=0))

# Plot the generated data
plt.figure(figsize=(8, 5))
plt.scatter(xx_np, y_obs_np, label="Observed)", color="black")
plt.plot(xx_np, y_true_np, label="True function", linestyle="dashed", color="black")
plt.plot(xx_np, ols_fit, label="nonrobust fit")
plt.plot(xx_np, pseudo_huber_fit, label="pseudo-Huber fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Robust Linear Regression from Langevin M-estimators")
plt.show()
