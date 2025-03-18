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
    """
    Computes the Huber influence function (psi).

    Args:
        x (array): Residual values.
        delta (float): Huber threshold.

    Returns:
        array: Huber influence function values.
    """
    return jnp.where(jnp.abs(x) <= delta, x, delta * jnp.sign(x))

def tukey_bisquare_psi(x, c=1):
    """
    Tukey's bisquare influence function (psi function).

    Args:
        x (array): Residual values.
        c (float): Tukey’s bisquare tuning constant (default = 4.685).

    Returns:
        array: Tukey bisquare psi values.
    """
    return jnp.where(jnp.abs(x) <= c, x * (1 - (x / c) ** 2) ** 2, 0)