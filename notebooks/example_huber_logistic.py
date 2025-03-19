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
from moulax.utils import pearson_residuals, make_fit_grad_fn, make_score_fun, identity_fisher, make_normal_prior
from moulax.psifun import ols_psi, pseudo_huber_psi, huber_psi, tukey_bisquare_psi


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
true_b = 1.5  # Some bias term

# Compute true probabilities using inverse logit
def sigmoid(eta):
    return 1 / (1 + jnp.exp(-eta))

eta_true = true_a * xx + true_b
p_true = sigmoid(eta_true)  # Probabilities

# Sample binomial responses (n=10 trials per observation)
n_trials = 1
key, subkey = random.split(key)
Y = random.binomial(subkey, n=n_trials, p=p_true)

# Add outliers
Y = Y.at[-int(N/20):].set(0)

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
    return fitted * (1 - fitted) + 1E-10 #avoid overflow


# -

# # Sample

# +
# %%time 
n_samples = xx.shape[0]
nonrobust_grad = make_score_fun(
    xx, Y,
    influence_fn=ols_psi, #
    residual_fn=pearson_residuals, #
    fit_fn=binomial_fit, 
    var_fn=binomial_variance,
    fit_grad_fn=None, # yes)
    average=True
)
normal_prior = make_normal_prior(jnp.array([0, 0]), 10.0 * n_samples)
def nonrobust_post_grad(theta, overdispersion):
    return nonrobust_grad(theta, overdispersion) + normal_prior(theta)

nonrobust_fisher = make_fisher_matrix(nonrobust_grad)

init_theta = jnp.array([0.0, 0.0])  

nonrobust_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta,
    grad_fn=nonrobust_post_grad, return_grad=True, fisher_func=nonrobust_fisher, n_samples=n_samples
)

# -

az.plot_trace(nonrobust_samples)
plt.tight_layout()

# +
# %%time 
n_samples = xx.shape[0]
huber_c = 2
pseudohuber_grad = make_score_fun(
    xx, Y,
    influence_fn=lambda theta: huber_psi(theta, huber_c), #
    residual_fn=pearson_residuals, #
    fit_fn=binomial_fit, 
    var_fn=binomial_variance,
    fit_grad_fn=None, # yes)
    average=True
)
normal_prior = make_normal_prior(jnp.array([0, 0]), 10.0 * n_samples)
def pseudohuber_post_grad(theta, overdispersion):
    return pseudohuber_grad(theta, overdispersion) + normal_prior(theta)

pseudohuber_fisher = make_fisher_matrix(pseudohuber_grad)

init_theta = jnp.array([0.1, 0.0])  

pseudohuber_samples = preconditioned_ULA(
    step,
    num_samples=10_000,
    step_size=0.1,
    fisher_updates=10,
    init_theta=init_theta,
    grad_fn=pseudohuber_post_grad,
    return_grad=True,
    fisher_func=nonrobust_fisher,
    n_samples=n_samples
)

# -

az.plot_trace(pseudohuber_samples)
plt.tight_layout()

# +
# # %%time 
# n_samples = xx.shape[0]
# tukey_c = 5.0
# tukey_grad = make_score_fun(
#     xx, Y,
#     influence_fn=lambda theta: tukey_bisquare_psi(theta, tukey_c), #
#     residual_fn=pearson_residuals, #
#     fit_fn=binomial_fit, 
#     var_fn=binomial_variance,
#     fit_grad_fn=None, # yes)
#     average=True
# )
# normal_prior = make_normal_prior(jnp.array([0, 0]), 10.0 * n_samples)
# def tukey_post_grad(theta, overdispersion):
#     return tukey_grad(theta, overdispersion) + normal_prior(theta)


# init_theta = jnp.array([2.0, 1.5])  

# tukey_samples = preconditioned_ULA(
#     step,
#     num_samples=10_000,
#     step_size=0.1,
#     fisher_updates=10,
#     init_theta=init_theta,
#     grad_fn=tukey_post_grad,
#     return_grad=True,
#     fisher_func=nonrobust_fisher,
#     n_samples=n_samples
# )


# +
# az.plot_trace(tukey_samples)
# plt.tight_layout()
# -

true_a, true_b

# +
fig, axes = plt.subplots(1,4, figsize=(14,3))

rr = jnp.linspace(-2,2,100)

axes[0].plot(rr, rr, label="OLS")
axes[0].plot(rr, huber_psi(rr, 0.12), label="QuasiHuber")
axes[0].plot(rr, tukey_bisquare_psi(rr, 1.0), label="Tukey")
axes[0].set_ylim([-1/3,1/3])
axes[0].legend()
axes[0].set_title("influence functions")

scaled_residuals = pearson_residuals(Y, p_true, binomial_variance(p_true))
axes[1].hist(scaled_residuals)
axes[1].set_title("residuals at true value")
axes[2].hist(huber_psi(scaled_residuals, huber_c))
axes[2].set_title("Huber residuals at true value")
axes[3].hist(tukey_bisquare_psi(scaled_residuals , 6.0))
axes[3].set_title("Tukey residuals at true value")

# +
theta_mean = nonrobust_samples.sel(draw=slice(1001, None)).posterior["theta"].mean(dim=("chain", "draw")).values
ols_fit = binomial_fit(xx, theta_mean)

theta_mean = pseudohuber_samples.sel(draw=slice(1001, None)).posterior["theta"].mean(dim=("chain", "draw")).values
pseudo_huber_fit = binomial_fit(xx, theta_mean)
# pseudo_huber_fit = binomial_fit(xx, last_valid_theta)
# -

# Plot the generated data
plt.figure(figsize=(8, 5))
plt.scatter(xx_np, Y, label="Observed)", color="black", alpha=.5)
plt.plot(xx_np, p_true, label="True function", linestyle="dashed", color="black")
plt.plot(xx_np, ols_fit, label="nonrobust")
plt.plot(xx_np, pseudo_huber_fit, label="Huber")
# plt.plot(xx_np, binomial_fit(xx_np, thetas[-1]))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quasi-Bayesian Robust Logistic Regression from Langevin M-estimators")
plt.show()

# +
# Create figure with two subplots (one for 'a', one for 'b')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Extract posterior samples for 'a' and 'b'
theta_nonrobust = nonrobust_samples.posterior["theta"].values  # Convert to NumPy array if needed
theta_pseudohuber = pseudohuber_samples.posterior["theta"].values

# Plot posterior distribution for 'a'
az.plot_posterior(theta_nonrobust[..., 0], ax=axes[0], label="Non-Robust", color="blue")
az.plot_posterior(theta_pseudohuber[..., 0], ax=axes[0], label="Pseudo-Huber", color="orange")
axes[0].axvline(true_a, color='red', linestyle="--", label="True a")
axes[0].set_title("Posterior of a")
axes[0].legend()

# Plot posterior distribution for 'b'
az.plot_posterior(theta_nonrobust[..., 1], ax=axes[1], label="Non-Robust", color="blue")
az.plot_posterior(theta_pseudohuber[..., 1], ax=axes[1], label="Pseudo-Huber", color="orange")
axes[1].axvline(true_b, color='red', linestyle="--", label="True b")
axes[1].set_title("Posterior of b")
axes[1].legend()

# Show the figure
plt.tight_layout()
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Extract posterior samples for 'a' and 'b' from both samplers
a_nonrobust = theta_nonrobust[..., 0].flatten()
b_nonrobust = theta_nonrobust[..., 1].flatten()
a_pseudohuber = theta_pseudohuber[..., 0].flatten()
b_pseudohuber = theta_pseudohuber[..., 1].flatten()

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot contour for non-robust sampler
sns.kdeplot(x=a_nonrobust, y=b_nonrobust, levels=5, color="blue", linestyle="solid", label="Non-Robust", ax=ax)

# Plot contour for pseudo-Huber sampler
sns.kdeplot(x=a_pseudohuber, y=b_pseudohuber, levels=5, color="orange", linestyle="dashed", label="Pseudo-Huber", ax=ax)

# Plot true value as an 'X' marker
ax.scatter(true_a, true_b, color='red', marker='x', s=100, label="True Value")

# Labels and legend
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_title("Posterior Distribution of (a, b)")
ax.legend()

# Show plot
plt.show()

