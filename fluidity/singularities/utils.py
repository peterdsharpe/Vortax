

def smoothed_inv(x: float, radius: float) -> float:
    """
    Approximates 1/x with a function that smoothly approaches 0 as x approaches 0.

    This function provides a regularized version of the inverse distance function,
    which is useful for avoiding singularities in physical simulations.

    Args:
        x: Distance value
        radius: Smoothing radius parameter

    Returns:
        Smoothed inverse distance value
    """
    return x / (x**2 + radius**2)
