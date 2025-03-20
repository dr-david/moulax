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
from tqdm import tqdm


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


# +
# def preconditioned_ULA(
#     step_fun=None,
#     num_samples=1000,
#     step_size=0.1,
#     fisher_updates=1,
#     key=jax.random.PRNGKey(42),
#     init_theta=None,
#     grad_fn=None,
#     fisher_func=None,
#     return_grad=False,
#     n_samples=1.0,
#     fit_fn=None, 
#     var_fn=None,
#     x=None,
#     y=None,
#     residual_fn=None, 
#     psi_fun=None,
#     overdisp_fn = None,
    
# ):
#     """
#     Runs Preconditioned Unadjusted Langevin Algorithm (P-ULA) and returns an ArviZ InferenceData object.

#     Args:
#         step_fun (function): Function to compute the PULA step.
#         num_samples (int): Number of MCMC iterations.
#         step_size (float): Step size for Langevin updates.
#         fisher_updates (int): How often to recompute the Fisher matrix.
#         key (jax.random.PRNGKey): Random key for JAX.
#         init_theta (array): Initial parameter vector.
#         grad_fn (function): Gradient function that depends only on theta.
#         return_grad (bool): If True, return gradient values at each step.

#     Returns:
#         arviz.InferenceData: MCMC samples in ArviZ format.
#     """
#     if init_theta is None:
#         raise ValueError("Must provide an initial theta vector.")
#     if grad_fn is None:
#         raise ValueError("Must provide a gradient function (grad_fn).")
#     if step_fun is None:
#         raise ValueError("Must provide a step function (step_fun).")
#     if fisher_func is None: # TODO this is not the fisher, it takes the prior into account !
#         fisher_func = make_fisher_matrix(grad_fn)

#     # Initialize parameters
#     theta = jnp.array(init_theta)
#     overdispersion = 1.0
#     theta_samples = []
#     grad_samples = [] if return_grad else None  # Store gradients if needed
#     overdispersion_samples = []

#     # JIT compile functions
#     fisher_func = jax.jit(fisher_func)
#     grad_fn = jax.jit(grad_fn)
#     step_fun = jax.jit(step_fun, static_argnames=["grad_fn", "return_grad"]) #why is it slow if I uncomment this ??
#     if overdisp_fn is not None:
#         fit_fn = jax.jit(fit_fn)
#         var_fn = jax.jit(var_fn)
#         residual_fn = jax.jit(residual_fn)
#         overdisp_fn = jax.jit(overdisp_fn, static_argnames=["psi_fun"])
    

#     # Iterate and sample 
#     for i in tqdm(range(num_samples)):
#         if i % fisher_updates == 0:
#             fisher_matrix = fisher_func(theta, overdispersion)
#             fisher_inv = jnp.linalg.inv(fisher_matrix)
#             sqrt_fisher_inv = jnp.linalg.cholesky(fisher_inv)
#             # Check for NaNs in Cholesky decomposition
#             if jnp.any(jnp.isnan(sqrt_fisher_inv)):
#                 warnings.warn(f"Cholesky decomposition resulted in NaNs at step {i}.", RuntimeWarning)

#         # Perform one step of P-ULA
#         if return_grad:
#             theta, overdispersion, key, grad_vec = step_fun(
#                 theta,
#                 overdispersion,
#                 key,
#                 step_size,
#                 fisher_inv,
#                 sqrt_fisher_inv,
#                 grad_fn,
#                 return_grad=True,
#                 n_samples=n_samples
#                 )
#             grad_samples.append(grad_vec)  # Store gradient
#         else:
#             theta, overdispersion, key = theta, overdispersion, key, grad_vec = step_fun(
#                 theta,
#                 overdispersion,
#                 key,
#                 step_size,
#                 fisher_inv,
#                 sqrt_fisher_inv,
#                 grad_fn,
#                 return_grad=False,
#                 n_samples=n_samples
#                 )
#         # Store samples
#         theta_samples.append(theta)

#         # update overdispersion
#         if overdisp_fn is not None:
#             fitted_vals = fit_fn(x, theta)  # Compute fitted values
#             variance_vals = var_fn(fitted_vals)  # Compute variance estimates
#             residual_vals = residual_fn(y, fitted_vals, variance_vals)  # Compute residuals
            
#             overdispersion = overdisp_fn(residual_vals, psi_fun, 2.0) # compute overdispersion 
#         overdispersion_samples.append(overdispersion)

#     theta_samples = jnp.array(theta_samples)

#     # Convert to ArviZ InferenceData
#     inference_dict = {
#         "posterior": {"theta": theta_samples[None, :, :]},  # Add chain dimension (single chain)
#         "coords": {"parameter": ["param_" + str(i) for i in range(theta_samples.shape[1])]},
#         "dims": {"theta": ["chain", "draw", "parameter"]},
#     }

#     # Add gradient values if requested
#     if return_grad:
#         grad_samples = jnp.array(grad_samples)
#         inference_dict["posterior"]["grad"] = grad_samples[None, :, :]  # Add chain dimension
#         inference_dict["dims"]["grad"] = ["chain", "draw", "parameter"]
    
#     overdispersion_samples = jnp.array(overdispersion_samples)

#     inference_dict["posterior"]["overdispersion"] = overdispersion_samples[None, :]  # Add chain dimension
#     inference_dict["dims"]["overdispersion"] = ["chain", "draw"]

#     return az.from_dict(**inference_dict)
# -

# # Sample

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
    return ols_grad(theta, overdispersion) + normal_prior(theta)

init_theta = jnp.array([0.0, 0.0])  


nonrobust_samples = preconditioned_ULA(
    step_fun=step,
    num_samples=100_000,
    step_size=0.1,
    fisher_updates=1,
    key=jax.random.PRNGKey(42),
    init_theta=init_theta,
    grad_fn=ols_post_grad,
    fisher_func=None,
    return_grad=True,
    n_samples=n_samples,
    fit_fn=linear_fit, 
    var_fn=constant_var,
    x=xx,
    y=y_obs,
    residual_fn=pearson_residuals, 
    psi_fun=ols_psi,
    overdisp_fn=overdispersion_mom
)

# -

az.plot_trace(nonrobust_samples)
plt.tight_layout()

# +
# %%time 
n_samples = xx.shape[0]

huber_c = 0.8
pseudohuber_grad = make_score_fun(
    xx, y_obs,
    influence_fn=lambda x: huber_psi(x, huber_c), #
    residual_fn=pearson_residuals, #
    fit_fn=linear_fit, # eg linear_fit
    var_fn=constant_var, # constant_var
    fit_grad_fn=None, # yes)
    average=True
)
init_theta = jnp.array([0.0, 0.0])  

normal_prior = make_normal_prior(jnp.array([0, 0]), 2.0*n_samples)
def pseudohuber_post_grad(theta, overdispersion):
    return pseudohuber_grad(theta, overdispersion) + normal_prior(theta)

# pseudo_huber_samples = preconditioned_ULA(
#     step, num_samples=10_000, step_size=0.1, fisher_updates=1, init_theta=init_theta, grad_fn=pseudohuber_post_grad, return_grad=True, n_samples=n_samples
# )

###

pseudo_huber_samples = preconditioned_ULA(
    step_fun=step,
    num_samples=20_000,
    step_size=0.1,
    fisher_updates=1,
    key=jax.random.PRNGKey(42),
    init_theta=init_theta,
    grad_fn=pseudohuber_post_grad,
    fisher_func=None,
    return_grad=True,
    n_samples=n_samples,
    fit_fn=linear_fit, 
    var_fn=constant_var,
    x=xx,
    y=y_obs,
    residual_fn=pearson_residuals, 
    psi_fun=lambda x: pseudo_huber_psi(x, huber_c),
    overdisp_fn=overdispersion_mom
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
    return tukey_grad(theta, overdispersion) + normal_prior(theta)

init_theta = jnp.array([0.5, 0.0])  
# init_theta = jnp.array([10.0, 10.0])  # faraway init !

ols_fisher = make_fisher_matrix(ols_grad)
# tukey_samples = preconditioned_ULA(
#     step, num_samples=20_000, step_size=0.1, fisher_updates=100, init_theta=init_theta, grad_fn=posterior_grad, return_grad=True, fisher_func=ols_fisher, n_samples=n_samples
# )

tukey_samples = preconditioned_ULA(
    step_fun=step,
    num_samples=20_000,
    step_size=0.1,
    fisher_updates=1,
    key=jax.random.PRNGKey(42),
    init_theta=init_theta,
    grad_fn=posterior_grad,
    fisher_func=ols_fisher,
    return_grad=True,
    n_samples=n_samples,
    fit_fn=linear_fit, 
    var_fn=constant_var,
    x=xx,
    y=y_obs,
    residual_fn=pearson_residuals, 
    psi_fun=lambda theta: tukey_bisquare_psi(theta, tukey_c), #
    overdisp_fn=overdispersion_mom
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

scaled_residuals = pearson_residuals(y_obs, y_true, constant_var(y_true)) # NEEED OVERDISP SCALING
axes[1].hist(
    scaled_residuals / nonrobust_samples.posterior["overdispersion"].mean().values
)
axes[1].set_title("residuals at true value")
axes[2].hist(huber_psi(
    scaled_residuals / pseudo_huber_samples.posterior["overdispersion"].mean().values
    , huber_c))
axes[2].set_title("Huber residuals at true value")
axes[3].hist(tukey_bisquare_psi(
    scaled_residuals / tukey_samples.posterior["overdispersion"].mean().values
    , tukey_c))
axes[3].set_title("Tukey residuals at true value")

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

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Extract posterior samples for 'a' and 'b'
theta_nonrobust = nonrobust_samples.sel(draw=slice(1_001)).posterior["theta"].values  # Convert to NumPy array if needed
theta_pseudohuber = pseudo_huber_samples.sel(draw=slice(1_001)).posterior["theta"].values
theta_tukey = tukey_samples.sel(draw=slice(1_001)).posterior["theta"].values

# Extract posterior samples for 'a' and 'b' from both samplers
a_nonrobust = theta_nonrobust[..., 0].flatten()
b_nonrobust = theta_nonrobust[..., 1].flatten()
a_pseudohuber = theta_pseudohuber[..., 0].flatten()
b_pseudohuber = theta_pseudohuber[..., 1].flatten()
a_tukey = theta_tukey[..., 0].flatten()
b_tukey = theta_tukey[..., 1].flatten()

# Create figure and axis
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

# Plot contour for non-robust sampler
ax = axes[0]
sns.kdeplot(x=a_nonrobust, y=b_nonrobust, levels=5, color="blue", linestyle="solid", label="Non-Robust", ax=ax)
ax.set_title("Nonrobust posterior")
ax.set_xlabel("slope")
ax.set_ylabel("intercept")
ax.scatter(true_a, true_b, color='red', marker='x', s=100, label="True Value")
ax.legend()


# Plot contour for pseudo-Huber sampler
ax = axes[1]
sns.kdeplot(x=a_pseudohuber, y=b_pseudohuber, levels=5, color="orange", linestyle="dashed", label="Pseudo-Huber", ax=ax)
ax.set_title("Huber quasiposterior")
ax.set_xlabel("slope")
ax.set_ylabel("intercept")
ax.scatter(true_a, true_b, color='red', marker='x', s=100, label="True Value")
ax.legend()

# Plot contour for pseudo-Huber sampler
ax = axes[2]
sns.kdeplot(x=a_tukey, y=b_tukey, levels=5, color="green", linestyle="dashed", label="Tukey", ax=ax)
ax.set_title("Tukey quasiposterior")
ax.set_xlabel("slope")
ax.set_ylabel("intercept")
ax.scatter(true_a, true_b, color='red', marker='x', s=100, label="True Value")
ax.legend()


# Show plot
fig.tight_layout()
fig.show()

