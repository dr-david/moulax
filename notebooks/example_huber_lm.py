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
from moulax.utils import pearson_residuals, make_fit_grad_fn, make_score_fun, identity_fisher, make_normal_prior, overdispersion_mom
from moulax.psifun import ols_psi, pseudo_huber_psi, huber_psi, tukey_bisquare_psi


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

## We fix the scale for all data
overdispersion_fixed = 0.1

# +
# %%time 
n_samples = xx.shape[0]
ols_grad = make_score_fun(
    xx, y_obs,
    influence_fn=ols_psi, #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
    average=True
)

normal_prior = make_normal_prior(jnp.array([0, 0]), 2.0 * n_samples)
def ols_post_grad(theta, overdispersion):
    return ols_grad(theta, overdispersion * overdispersion_fixed) + normal_prior(theta)

init_theta = jnp.array([0.0, 0.0])  

nonrobust_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=ols_post_grad, return_grad=True, n_samples=n_samples
)

# -

az.plot_trace(nonrobust_samples)
plt.tight_layout()

# +
# %%time 
n_samples = xx.shape[0]

huber_c = 0.2
pseudohuber_grad = make_score_fun(
    xx, y_obs,
    influence_fn=lambda x: pseudo_huber_psi(x, huber_c), #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
    average=True
)
init_theta = jnp.array([0.0, 0.0])  

normal_prior = make_normal_prior(jnp.array([0, 0]), 2.0*n_samples)
def pseudohuber_post_grad(theta, overdispersion):
    return pseudohuber_grad(theta, overdispersion * overdispersion_fixed) + normal_prior(theta)

pseudo_huber_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=pseudohuber_post_grad, return_grad=True, n_samples=n_samples
)

# -

az.plot_trace(pseudo_huber_samples)
plt.tight_layout()

# +
# %%time 
n_samples = xx.shape[0]
tukey_c = 3.0
tukey_grad = make_score_fun(
    xx, y_obs,
    influence_fn=lambda theta: tukey_bisquare_psi(theta, tukey_c), #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
    average=True,
)

normal_prior = make_normal_prior(jnp.array([0, 0]), 2.0*n_samples)

def posterior_grad(theta, overdispersion):
    return tukey_grad(theta, overdispersion * overdispersion_fixed) + normal_prior(theta)

init_theta = jnp.array([0.5, 0.0])  
# init_theta = jnp.array([10.0, 10.0])  # faraway init !

ols_fisher = make_fisher_matrix(ols_grad)
tukey_samples = preconditioned_ULA(
    step, num_samples=20_000, step_size=0.1, fisher_updates=100, init_theta=init_theta, grad_fn=posterior_grad, return_grad=True, fisher_func=ols_fisher, n_samples=n_samples
)

# -

az.plot_trace(tukey_samples)
plt.tight_layout()
# az.plot_trace(tukey_samples.sel(draw=slice(10_001, None)))

# +
fig, axes = plt.subplots(1,4, figsize=(14,3))

rr = jnp.linspace(-2,2,100)

axes[0].plot(rr, rr, label="OLS")
axes[0].plot(rr, pseudo_huber_psi(rr, 0.12), label="QuasiHuber")
axes[0].plot(rr, tukey_bisquare_psi(rr, 1.0), label="Tukey")
axes[0].set_ylim([-1/3,1/3])
axes[0].legend()
axes[0].set_title("influence functions")

scaled_residuals = (y_obs - y_true) / overdispersion_fixed
axes[1].hist(scaled_residuals)
axes[1].set_title("residuals at true value")
axes[2].hist(pseudo_huber_psi(scaled_residuals , huber_c))
axes[2].set_title("Huber residuals at true value")
axes[3].hist(tukey_bisquare_psi(scaled_residuals , tukey_c))
axes[3].set_title("Tukey residuals at true value")
# -



# +
true_theta = jnp.array([true_a, true_b])
far_theta = jnp.array([-10.0, -10.0])
print(f"Tukey loss grad at true theta: {tukey_grad(true_theta, 1.0)}")
print(f"Tukey posterior grad at true theta: {posterior_grad(true_theta, 1.0)}")

print(f"Tukey loss grad at far theta: {tukey_grad(far_theta, 1.0)}")
print(f"Prior grad at far theta: {normal_prior(far_theta)}")
print(f"Tukey posterior grad at far theta: {posterior_grad(far_theta, 1.0)}")

ols_fisher = make_fisher_matrix(ols_grad)
tukey_fisher = make_fisher_matrix(tukey_grad)

print(f"\nOLS inverse Fisher at true theta:\n{jnp.linalg.inv(ols_fisher(true_theta, 1.0))}")
print(f"\nTukey inverse Fisher at true theta:\n{jnp.linalg.inv(tukey_fisher(true_theta, 1.0))}")

print(f"\nOLS inverse Fisher at far theta:\n{jnp.linalg.inv(ols_fisher(far_theta, 1.0))}")
print(f"\nTukey inverse Fisher at far theta:\n{jnp.linalg.inv(tukey_fisher(far_theta, 1.0))}")





# +
theta_mean = nonrobust_samples.sel(draw=slice(1_001)).posterior["theta"].mean(dim=("chain", "draw")).values
ols_fit = linear_fit(xx, theta_mean)

theta_mean = pseudo_huber_samples.sel(draw=slice(1_001)).posterior["theta"].mean(dim=("chain", "draw")).values
pseudo_huber_fit = linear_fit(xx, theta_mean)

theta_mean = tukey_samples.sel(draw=slice(1_001)).posterior["theta"].mean(dim=("chain", "draw")).values
tukey_fit = linear_fit(xx, theta_mean)

# +
# Plot the generated data
plt.figure(figsize=(8, 5))
plt.scatter(xx_np, y_obs_np, label="Observed)", color="black", alpha=.5)
plt.plot(xx_np, y_true_np, label="True function", linestyle="dashed", color="black")
plt.plot(xx_np, ols_fit, label="square loss")
plt.plot(xx_np, pseudo_huber_fit, label="pseudo-Huber fit")
plt.plot(xx_np, tukey_fit, label="Tukey fit")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quasi-Bayesian Robust Linear Regression from Langevin M-estimators")
plt.show()
