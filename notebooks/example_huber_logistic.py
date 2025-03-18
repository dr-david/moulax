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
def nonrobust_post_grad(theta):
    return nonrobust_grad(theta) + normal_prior(theta)

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
pseudohuber_grad = make_score_fun(
    xx, Y,
    influence_fn=lambda theta: huber_psi(theta, 2), #
    residual_fn=pearson_residuals, #
    fit_fn=binomial_fit, 
    var_fn=binomial_variance,
    fit_grad_fn=None, # yes)
    average=True
)
normal_prior = make_normal_prior(jnp.array([0, 0]), 10.0 * n_samples)
def pseudohuber_post_grad(theta):
    return pseudohuber_grad(theta) + normal_prior(theta)

pseudohuber_fisher = make_fisher_matrix(pseudohuber_grad)

init_theta = jnp.array([0.0, 0.0])  

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
fig, axes = plt.subplots(1,2, figsize=(10,3))

theta_mean =  pseudohuber_samples.posterior["theta"].mean(dim=("chain", "draw")).values
fitted_vals = binomial_fit(xx, theta_mean)
variance_vals = binomial_variance(fitted_vals)
residual_vals = pearson_residuals(Y, fitted_vals, variance_vals) 
influence_vals = huber_psi(residual_vals, 2.0)

print(influence_vals.mean())
print(pseudohuber_grad(theta_mean))


axes[0].hist(residual_vals, bins=30)
axes[0].set_title("Pearson residuals")
axes[1].hist(influence_vals, bins=30)
axes[1].set_title("Huberized Pearson residuals")

fig.show()

# +
from moulax.utils import robust_mean

print(f"mean square pearson residuals: {jnp.sum(residual_vals**2) / (n_samples -2)}")

## ddof = sum of the weights - p_params. the weights * residuals = psi_fun(residuals)

ddof = jnp.sum(influence_vals / residual_vals) -2
print(f"mean square huberized pearson residuals: {jnp.sum(influence_vals**2) / ddof}")
print(f"huberized mean square pearson residuals: {robust_mean(influence_vals**2)}")



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

az.plot_posterior(nonrobust_samples.posterior["theta"])
az.plot_posterior(pseudohuber_samples.posterior["theta"])
