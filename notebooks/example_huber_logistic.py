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
from moulax.sampling import step, preconditioned_ULA, make_fisher_matrix, make_fisher_matrix_outer
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
xx = jnp.linspace(-2, 2, N)  # Now centered for logit transformation

# Define true function parameters
true_a = 2.0
true_b = -0.5  # Some bias term

# Compute true probabilities using inverse logit
def sigmoid(eta):
    return 1 / (1 + jnp.exp(-eta))

eta_true = true_a * xx + true_b
p_true = sigmoid(eta_true)  # Probabilities

# Sample binomial responses (n=10 trials per observation)
n_trials = 1
key, subkey = random.split(key)
Y = random.binomial(subkey, n=n_trials, p=p_true)

# Convert to NumPy for plotting
xx_np = jnp.array(xx)
Y_np = jnp.array(Y) / n_trials  # Convert to proportion scale


# -

# # Define the gradients
#
# We define all the functions necessary to get the gradients.

# +
# Define binomial fit function
def binomial_fit(x, theta):
    """Computes the probability p using inverse logit transformation."""
    a, b = theta
    eta = a * x + b
    return sigmoid(eta)

# Define variance function for binomial
def binomial_variance(fitted):
    """Computes binomial variance: p * (1 - p)."""
    return fitted * (1 - fitted) + 1E-6 #avoid overflow


# -

# # Sample

def identity_fisher(theta):
    return jnp.eye(theta.shape[0])


# +
# %%time 
nonrobust_grad = make_score_fun(
    xx, Y,
    influence_fn=ols_psi, #
    residual_fn=pearson_residuals, #
    fit_fn=binomial_fit, 
    var_fn=binomial_variance,
    fit_grad_fn=None, # yes)
)
nonrobust_fisher = make_fisher_matrix(nonrobust_grad)

init_theta = jnp.array([0.0, 0.0])  

nonrobust_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=nonrobust_grad, return_grad=True, fisher_func=identity_fisher
)

# -

az.plot_trace(nonrobust_samples)
plt.tight_layout()



# +
# %%time 
def infl_fun(x):
    return pseudo_huber_psi(x, 3)

pseudohuber_grad = make_score_fun(
    xx, Y,
    influence_fn=infl_fun, #
    residual_fn=pearson_residuals, #
    fit_fn=binomial_fit, # eg linear_fit
    var_fn=binomial_variance, # constant_var
    fit_grad_fn=None, # yes)
)
init_theta = jnp.array([0.0, 0.0])  

pseudo_huber_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=pseudohuber_grad, return_grad=True, fisher_func=identity_fisher
)

# -

pseudohuber_fisher = make_fisher_matrix(pseudohuber_grad)
nonrobust_fisher = make_fisher_matrix(nonrobust_grad)

# +
# Extract posterior samples for theta
theta_samples = pseudo_huber_samples.posterior["theta"].values  # Shape: (chains, draws, parameters)

# Find last valid index along the draw axis
valid_mask = ~jnp.isnan(theta_samples)  # True where values are not NaN

# Create an index array that matches theta_samples shape
index_array = jnp.arange(theta_samples.shape[1]).reshape(1, -1, 1)  # Reshape to (1, 100, 1)

# Replace invalid indices with -1
last_valid_index = jnp.max(jnp.where(valid_mask, index_array, -1), axis=1)  # Shape: (1, parameters)

# Retrieve the last non-NaN value for each parameter
last_valid_theta = jnp.take_along_axis(theta_samples, last_valid_index[:, None, :], axis=1).squeeze(1)[0]
last_valid_theta
# -

last_valid_theta = jnp.array([ 2.0, -0.5 ])

fitted_vals = binomial_fit(xx, last_valid_theta)
variance_vals = binomial_variance(fitted_vals)
residual_vals = pearson_residuals(Y, fitted_vals, variance_vals) 
influence_vals = pseudo_huber_psi(residual_vals, 3)
influence_vals_ols = ols_psi(residual_vals)

# +
fig, axes = plt.subplots(1,2, figsize=(10,3))

rr = jnp.linspace(-10,10,100)
axes[0].plot(rr, ols_psi(rr))
axes[0].plot(rr, pseudo_huber_psi(rr, 3))


# +
fig, axes = plt.subplots(1,3, figsize=(10,3))

axes[0].hist(residual_vals)
axes[1].hist(influence_vals)
axes[2].hist(influence_vals_ols)

fig.show()
# -

pseudohuber_grad(last_valid_theta)


fish_ols = nonrobust_fisher(last_valid_theta)
print(fish_ols)
print(jnp.linalg.inv(fish_ols))

fish_hub = pseudohuber_fisher(last_valid_theta[0])
print(fish_hub)
print(jnp.linalg.inv(fish_hub))

az.plot_trace(pseudo_huber_samples)

 nonrobust_samples.posterior["theta"].mean(dim=("chain", "draw")).values

# +
theta_mean = nonrobust_samples.posterior["theta"].mean(dim=("chain", "draw")).values
ols_fit = binomial_fit(xx, theta_mean)

theta_mean = pseudo_huber_samples.posterior["theta"].mean(dim=("chain", "draw")).values
pseudo_huber_fit = binomial_fit(xx, theta_mean)
# pseudo_huber_fit = binomial_fit(xx, last_valid_theta)
# -

last_valid_theta

# Plot the generated data
plt.figure(figsize=(8, 5))
plt.scatter(xx_np, Y, label="Observed)", color="black", alpha=.5)
plt.plot(xx_np, p_true, label="True function", linestyle="dashed", color="black")
plt.plot(xx_np, ols_fit, label="nonrobust fit")
plt.plot(xx_np, pseudo_huber_fit, label="pseudo-Huber fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Robust Logistic Regression from Langevin M-estimators")
plt.show()
