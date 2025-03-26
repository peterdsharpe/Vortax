import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Float
from fluidity.types import vec3
from fluidity.singularities.utils import smoothed_inv

eps = 1e-8


@jax.jit
def induced_velocity_vortex_filament(
    query_point: vec3,
    start_point: vec3,
    end_point: vec3,
    radius: float = eps,
    strength: float = 1.0,
) -> vec3:
    """
    Compute the velocity field induced by a vortex filament.

    Args:
        query_point: Point where velocity is evaluated (3D vector)
        start_point: Start point of vortex filament (3D vector)
        end_point: End point of vortex filament (3D vector)
        radius: Smoothing radius for singularity avoidance
        strength: Strength of the vortex filament

    Returns:
        Induced velocity vector at query point
    """
    # Vector from query point to start and end points
    a = start_point - query_point
    b = end_point - query_point

    # Precompute some quantities for speed
    a_dot_b = jnp.dot(a, b)
    a_cross_b = jnp.cross(a, b)
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    norm_a_inv = smoothed_inv(norm_a, radius)
    norm_b_inv = smoothed_inv(norm_b, radius)

    constant = (
        (1.0 / (4.0 * jnp.pi))
        * a_cross_b
        * (norm_a_inv + norm_b_inv)
        * smoothed_inv(norm_a * norm_b + a_dot_b, radius)
    )

    return constant * strength


@jax.jit
def induced_velocity_ring_vortex(
    query_point: vec3,
    ring_points: Float[vec3, " ring_points"],
    radius: float = eps,
) -> vec3:
    """
    Compute the velocity field induced by a ring vortex.

    Args:
        query_point: Point where velocity is evaluated (3D vector)
        ring_points: Array of points defining the ring vortex (Nx3 array)
        radius: Smoothing radius for singularity avoidance

    Returns:
        Induced velocity vector at query point
    """
    # Dynamically get the number of points from the jaxtyping annotation
    n_ring_points = ring_points.shape[0]

    # Vectorized computation for all segments
    def compute_segment_velocity(i):
        start_point = ring_points[i]
        end_point = ring_points[(i + 1) % n_ring_points]
        return induced_velocity_vortex_filament(
            query_point=query_point,
            start_point=start_point,
            end_point=end_point,
            radius=radius,
        )

    # Map the computation over all segments and sum the results
    velocities = jax.vmap(compute_segment_velocity)(jnp.arange(n_ring_points))
    return jnp.sum(velocities, axis=0)


if __name__ == "__main__":
    # Test case 1: Point above a horizontal filament
    start_point = jnp.array([0.0, 0.0, 0.0])
    end_point = jnp.array([1.0, 0.0, 0.0])
    query_point = jnp.array([0.5, 0.0, 1.0])

    velocity = induced_velocity_vortex_filament(
        query_point=query_point, start_point=start_point, end_point=end_point
    )

    print("Test case 1: Point above horizontal filament")
    print(f"{np.array(velocity) = }")
    print("Expected direction: negative y-axis")

    # Test case 2: Point further away should have smaller velocity magnitude
    query_point_far = jnp.array([0.5, 0.0, 2.0])
    velocity_far = induced_velocity_vortex_filament(
        query_point=query_point_far, start_point=start_point, end_point=end_point
    )

    print("\nTest case 2: Point further away")
    print(f"{np.array(velocity) = }, magnitude: {jnp.linalg.norm(velocity) = }")
    print(f"{np.array(velocity_far) = }, magnitude: {jnp.linalg.norm(velocity_far) = }")
    print(f"{jnp.linalg.norm(velocity_far) / jnp.linalg.norm(velocity) = }")

    # Test case 3: With specified strength
    velocity_strong = induced_velocity_vortex_filament(
        query_point=query_point,
        start_point=start_point,
        end_point=end_point,
        strength=2.0,
    )

    print("\nTest case 3: With doubled strength")
    print(f"{np.array(velocity) = }")
    print(f"{np.array(velocity_strong) = }")
    print(f"{jnp.linalg.norm(velocity_strong) / jnp.linalg.norm(velocity) = }")
    # Check individual components safely
    for i, component in enumerate(["x", "y", "z"]):
        if abs(velocity[i]) > 1e-10:  # Avoid division by near-zero values
            print(
                f"{component}-component ratio: {velocity_strong[i] / velocity[i]:.2f}"
            )
        else:
            print(f"{component}-component: near zero in original velocity")

    # Test case 4: Ring vortex
    print("\nTest case 4: Ring vortex")

    # Define the vertices of a panel in the xz-plane
    vertices = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
    )

    # Define a query point above the panel
    query_point = jnp.array([0.0, 0.0, 1.0])

    # Calculate induced velocity from each edge of the triangle
    v_edges = [
        induced_velocity_vortex_filament(
            query_point=query_point,
            start_point=vertices[i],
            end_point=vertices[(i + 1) % len(vertices)],
        )
        for i in range(len(vertices))
    ]

    # Total velocity is the sum of contributions from all edges
    v_total = sum(v_edges)

    print(f"{np.array(query_point) = }")
    print(f"{np.array(vertices) = }")
    print(f"{np.array(v_total) = }, {jnp.linalg.norm(v_total) = }")

    # Using the ring vortex function
    v_total_ring = induced_velocity_ring_vortex(
        query_point=query_point,
        ring_points=vertices,
        radius=eps,
    )

    print(f"{np.array(v_total_ring) = }, magnitude: {jnp.linalg.norm(v_total_ring) = }")
