import jax.numpy as jnp


def ols_psi(x):
    """Ordinary Least Squares (OLS) influence function (identity)."""
    return x

def pseudo_huber_psi(x, delta=0.12):
    """
    Computes the influence function (psi) for the Pseudo-Huber loss.

    Args:
        x (array): Residual values.
        delta (float): Pseudo-Huber threshold (default = 1.0).

    Returns:
        array: Pseudo-Huber influence function values.
    """
    return x / jnp.sqrt(1 + (x / delta) ** 2)

def huber_psi(x, delta=1.0):
    """Huber influence function."""
    return jnp.where(jnp.abs(x) <= delta, x, delta * jnp.sign(x))