import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd
from tqdm import tqdm
import arviz as az

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
        grad_fn: Function that computes gradients w.r.t. theta.

    Returns:
        Fisher Information Matrix (Jacobian of grad_fn).
    """
    def fisher_fn(theta):
        return -jacfwd(grad_fn)(theta) + 1e-6 * jnp.eye(theta.shape[0])  # Ensure positive definiteness
    
    return fisher_fn

def step(theta, s, key, step_size, fisher_inv, sqrt_fisher_inv, grad_fn):
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

    Returns:
        Updated (theta, s) and updated key.
    """
    key, noise_key1 = jax.random.split(key, 2)

    # Compute gradient of log-likelihood
    grad_vec = grad_fn(theta)  # Now depends only on theta

    # Preconditioned Gradient Update
    preconditioned_grad = fisher_inv @ grad_vec  

    # Langevin Noise
    noise_vec = jax.random.normal(noise_key1, theta.shape)
    noise_term = sqrt_fisher_inv @ noise_vec

    # Langevin Update for parameters
    update = (step_size / 2) * preconditioned_grad + jnp.sqrt(step_size) * noise_term * jnp.sqrt(s)
    theta = theta + update  # Update parameters

    return theta, s, key

def preconditioned_ULA(
    step_fun=None,
    num_samples=1000,
    step_size=0.1,
    fisher_updates=1,
    key=jax.random.PRNGKey(42),
    init_theta=None,
    grad_fn=None,
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

    Returns:
        arviz.InferenceData: MCMC samples in ArviZ format.
    """
    if init_theta is None:
        raise ValueError("Must provide an initial theta vector.")
    if grad_fn is None:
        raise ValueError("Must provide a gradient function (grad_fn).")
    if step_fun is None:
        raise ValueError("Must provide a step function (step_fun).")

    fisher_func = make_fisher_matrix(grad_fn)

    # Initialize parameters
    theta = jnp.array(init_theta)
    s = 1.0
    theta_samples = []

    # JIT compile functions
    fisher_func = jax.jit(fisher_func)
    grad_fn = jax.jit(grad_fn)
    step_fun = jax.jit(step_fun, static_argnames=["grad_fn"])

    for i in tqdm(range(num_samples)):
        if i % fisher_updates == 0:
            fisher_matrix = fisher_func(theta)
            fisher_inv = jnp.linalg.inv(fisher_matrix)
            sqrt_fisher_inv = jnp.linalg.cholesky(fisher_inv)

        # Perform one step of P-ULA
        theta, s, key = step_fun(theta, s, key, step_size, fisher_inv, sqrt_fisher_inv, grad_fn)
        
        # Store samples
        theta_samples.append(theta)

    theta_samples = jnp.array(theta_samples)

    # Convert to ArviZ InferenceData
    return az.from_dict(
        posterior={"theta": theta_samples[None, :, :]},  # Add chain dimension (single chain)
        coords={"parameter": ["param_" + str(i) for i in range(theta_samples.shape[1])]},
        dims={"theta": ["chain", "draw", "parameter"]},
    )
