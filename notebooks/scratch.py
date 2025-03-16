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

# +
from mestim_mcmc.sampling import preconditioned_ULA
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mestim_mcmc.scores import f, quasibinomial_log_likelihood
from mestim_mcmc.sampling import step, preconditioned_ULA, make_gradient



# +
# %%time 
xx = jnp.arange(-20, 21)  # Time series
a_true, b_true = 0.2, 1.0  # True parameters
Y = jax.random.binomial(jax.random.PRNGKey(0), n=10, p=f(a_true, b_true, xx))  # Observed data
n_trials = jnp.full_like(Y, 10)  # Number of trials


def log_likelihood(theta):
    a, b = theta  # Assuming 2D parameters
    mu = f(a, b, xx)
    return quasibinomial_log_likelihood(Y, mu, n_trials, 1.0)
    
grad_fn = make_gradient(log_likelihood)
init_theta = jnp.array([0.0, 0.0])  # Initial values for a, b
theta_samples = preconditioned_ULA(
    step, num_samples=10_000, step_size=0.1, fisher_updates=10, init_theta=init_theta, grad_fn=grad_fn
)
# -

plt.plot(jnp.arange(theta_samples.shape[0]), theta_samples)
