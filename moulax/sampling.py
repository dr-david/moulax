import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd
from tqdm import tqdm
import arviz as az
import warnings

from .utils import robust_mean, overdispersion_mom

def make_gradient(log_likelihood_fn):
    """
    Returns a compiled gradient function that depends only on theta.
    
    Args:
        log_likelihood_fn: A function that computes the log-likelihood.

    Returns:
        A function that computes gradients w.r.t. theta.
    """
    def grad_fn(theta):
        return grad(log_likelihood_fn)(theta)

    return grad_fn

def make_fisher_matrix(grad_fn):
    """
    Computes the Fisher Information Matrix from the provided gradient function.

    Args:
        grad_fn: Function that computes gradients w.r.t. theta, according to some overdispersion value

    Returns:
        Fisher Information Matrix (Jacobian of grad_fn).
    """
    def fisher_fn(theta, overdispersion):
        grad_fn_s = lambda theta: grad_fn(theta, overdispersion)
        return -jacfwd(grad_fn_s)(theta) + 1e-6 * jnp.eye(theta.shape[0])  # Ensure positive definiteness
    
    return fisher_fn

def make_fisher_matrix_outer(grad_fn): #Deprecated
    """
    Computes the Fisher Information Matrix using the outer product of the gradient.

    Args:
        grad_fn: Function that computes gradients w.r.t. theta.

    Returns:
        fisher_fn: Function that computes Fisher Information Matrix at theta.
    """
    def fisher_fn(theta):
        grad_vals = grad_fn(theta)  # Compute gradient vector
        fisher_matrix = grad_vals[:, None] @ grad_vals[None, :]  # Outer product
        return fisher_matrix + 1e-6 * jnp.eye(theta.shape[0])  # Ensure positive definiteness

    return fisher_fn

def step(theta, overdispersion, key, step_size, fisher_inv, sqrt_fisher_inv, grad_fn, return_grad=False, n_samples=1.0):
    """
    Performs a single Langevin update step for theta.

    Args:
        theta: Parameter vector.
        s: Overdispersion parameter.
        key: Random key for JAX.
        step_size: Langevin step size.
        fisher_inv: Inverse Fisher Information Matrix.
        sqrt_fisher_inv: Cholesky decomposition of fisher_inv.
        grad_fn: Precomputed gradient function that depends only on theta.
        return_grad (bool): If True, return the gradient values at each step.

    Returns:
        Updated (theta, s), updated key, and optionally the gradient vector.
    """
    key, noise_key1 = jax.random.split(key, 2)

    # Compute gradient of log-likelihood
    grad_vec = grad_fn(theta, overdispersion)

    # Preconditioned Gradient Update
    preconditioned_grad = fisher_inv @ grad_vec  

    # Langevin Noise
    noise_vec = jax.random.normal(noise_key1, theta.shape)
    noise_term = sqrt_fisher_inv @ noise_vec

    # Langevin Update for parameters
    update = (step_size / 2) * preconditioned_grad + jnp.sqrt(step_size) * noise_term * jnp.sqrt(overdispersion) / jnp.sqrt(n_samples)
    theta = theta + update  # Update parameters

    if return_grad:
        return theta, overdispersion, key, grad_vec
    else:
        return theta, overdispersion, key
    

def preconditioned_ULA(
    step_fun=None,
    num_samples=1000,
    step_size=0.1,
    fisher_updates=1,
    key=jax.random.PRNGKey(42),
    init_theta=None,
    grad_fn=None,
    fisher_func=None,
    return_grad=False,
    n_samples=1.0,
    fit_fn=None, 
    var_fn=None,
    x=None,
    y=None,
    residual_fn=None, 
    psi_fun=None,
    overdisp_fn = None,
    
):
    """
    Runs Preconditioned Unadjusted Langevin Algorithm (P-ULA) and returns an ArviZ InferenceData object.

    Args:
        step_fun (function): Function to compute the PULA step.
        num_samples (int): Number of MCMC iterations.
        step_size (float): Step size for Langevin updates.
        fisher_updates (int): How often to recompute the Fisher matrix.
        key (jax.random.PRNGKey): Random key for JAX.
        init_theta (array): Initial parameter vector.
        grad_fn (function): Gradient function that depends only on theta.
        return_grad (bool): If True, return gradient values at each step.

    Returns:
        arviz.InferenceData: MCMC samples in ArviZ format.
    """
    if init_theta is None:
        raise ValueError("Must provide an initial theta vector.")
    if grad_fn is None:
        raise ValueError("Must provide a gradient function (grad_fn).")
    if step_fun is None:
        raise ValueError("Must provide a step function (step_fun).")
    if fisher_func is None: # TODO this is not the fisher, it takes the prior into account !
        fisher_func = make_fisher_matrix(grad_fn)

    # Initialize parameters
    theta = jnp.array(init_theta)
    overdispersion = 1.0
    theta_samples = []
    grad_samples = [] if return_grad else None  # Store gradients if needed
    overdispersion_samples = []

    # JIT compile functions
    fisher_func = jax.jit(fisher_func)
    grad_fn = jax.jit(grad_fn)
    step_fun = jax.jit(step_fun, static_argnames=["grad_fn", "return_grad"]) #why is it slow if I uncomment this ??
    if overdisp_fn is not None:
        fit_fn = jax.jit(fit_fn)
        var_fn = jax.jit(var_fn)
        residual_fn = jax.jit(residual_fn)
        overdisp_fn = jax.jit(overdisp_fn, static_argnames=["psi_fun"])
    

    # Iterate and sample 
    for i in tqdm(range(num_samples)):
        if i % fisher_updates == 0:
            fisher_matrix = fisher_func(theta, overdispersion)
            fisher_inv = jnp.linalg.inv(fisher_matrix)
            sqrt_fisher_inv = jnp.linalg.cholesky(fisher_inv)
            # Check for NaNs in Cholesky decomposition
            if jnp.any(jnp.isnan(sqrt_fisher_inv)):
                warnings.warn(f"Cholesky decomposition resulted in NaNs at step {i}.", RuntimeWarning)

        # Perform one step of P-ULA
        if return_grad:
            theta, overdispersion, key, grad_vec = step_fun(
                theta,
                overdispersion,
                key,
                step_size,
                fisher_inv,
                sqrt_fisher_inv,
                grad_fn,
                return_grad=True,
                n_samples=n_samples
                )
            grad_samples.append(grad_vec)  # Store gradient
        else:
            theta, overdispersion, key = theta, overdispersion, key, grad_vec = step_fun(
                theta,
                overdispersion,
                key,
                step_size,
                fisher_inv,
                sqrt_fisher_inv,
                grad_fn,
                return_grad=False,
                n_samples=n_samples
                )
        # Store samples
        theta_samples.append(theta)

        # update overdispersion
        if overdisp_fn is not None:
            fitted_vals = fit_fn(x, theta)  # Compute fitted values
            variance_vals = var_fn(fitted_vals)  # Compute variance estimates
            residual_vals = residual_fn(y, fitted_vals, variance_vals)  # Compute residuals
            
            overdispersion = overdisp_fn(residual_vals, psi_fun, 2.0) # compute overdispersion 
        overdispersion_samples.append(overdispersion)

    theta_samples = jnp.array(theta_samples)

    # Convert to ArviZ InferenceData
    inference_dict = {
        "posterior": {"theta": theta_samples[None, :, :]},  # Add chain dimension (single chain)
        "coords": {"parameter": ["param_" + str(i) for i in range(theta_samples.shape[1])]},
        "dims": {"theta": ["chain", "draw", "parameter"]},
    }

    # Add gradient values if requested
    if return_grad:
        grad_samples = jnp.array(grad_samples)
        inference_dict["posterior"]["grad"] = grad_samples[None, :, :]  # Add chain dimension
        inference_dict["dims"]["grad"] = ["chain", "draw", "parameter"]
    
    overdispersion_samples = jnp.array(overdispersion_samples)

    inference_dict["posterior"]["overdispersion"] = overdispersion_samples[None, :]  # Add chain dimension
    inference_dict["dims"]["overdispersion"] = ["chain", "draw"]

    return az.from_dict(**inference_dict)