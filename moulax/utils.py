# Define function f(a, b, t)
def f(a, b, t):
    return jnp.exp(a * t + b) / (1 + jnp.exp(a * t + b))