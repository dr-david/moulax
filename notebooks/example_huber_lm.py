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
from moulax.utils import pearson_residuals, make_fit_grad_fn, make_score_fun
from moulax.psifun import ols_psi, pseudo_huber_psi, huber_psi


# # Simulate data
#
# We simulate data from a gaussian linear model. We contaminate with two outliers with high leverage.

# +
# Set seed for reproducibility
key = random.PRNGKey(42)

# Generate x values
N = 100
xx = jnp.linspace(0,1,N)

# Define true function parameters
true_a = 1.0
true_b = 0.0

# Compute true y values
y_true = true_a * xx + true_b

# set outliers
y_outliers = y_true.copy()
y_outliers = y_outliers.at[-int(N/10):].set(0)

# Add Gaussian noise to create observed y values
noise_std = 1.0 / 20  # Standard deviation of noise
key, subkey = random.split(key)
y_obs = y_outliers + random.normal(subkey, shape=xx.shape) * noise_std


# Convert to NumPy for plotting
xx_np = jnp.array(xx)
y_obs_np = jnp.array(y_obs)
y_true_np = jnp.array(y_true)

# -

# # Define the gradients
#
# We define all the functions necessary to get the gradients.

# +
def linear_fit(x, theta):
    """Computes fitted values y = a*x + b."""
    a, b = theta
    return a * x + b

def constant_var(fitted):
    """Computes constant variance"""
    return jnp.ones_like(fitted)



# -

# # Sample

# +
# %%time 
ols_grad = make_score_fun(
    xx, y_obs,
    influence_fn=ols_psi, #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
)
init_theta = jnp.array([0.0, 0.0])  

nonrobust_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=ols_grad
)

# -

az.plot_trace(nonrobust_samples)

# +
# %%time 
pseudohuber_grad = make_score_fun(
    xx, y_obs,
    influence_fn=pseudo_huber_psi, #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
)
init_theta = jnp.array([0.0, 0.0])  

pseudo_huber_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=pseudohuber_grad
)

# -

az.plot_trace(pseudo_huber_samples)

# +
theta_mean = nonrobust_samples.posterior["theta"].mean(dim=("chain", "draw")).values
ols_fit = linear_fit(xx, theta_mean)

theta_mean = pseudo_huber_samples.posterior["theta"].mean(dim=("chain", "draw")).values
pseudo_huber_fit = linear_fit(xx, theta_mean)
# -

# Plot the generated data
plt.figure(figsize=(8, 5))
plt.scatter(xx_np, y_obs_np, label="Observed)", color="black", alpha=.5)
plt.plot(xx_np, y_true_np, label="True function", linestyle="dashed", color="black")
plt.plot(xx_np, ols_fit, label="nonrobust fit")
plt.plot(xx_np, pseudo_huber_fit, label="pseudo-Huber fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Robust Linear Regression from Langevin M-estimators")
plt.show()
