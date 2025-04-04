import jax
import numpy as np
from vortax.mesh import Mesh
from jaxtyping import Float, Array, Int
from vortax.singularities.utils import smoothed_inv
from vortax.types import vec3
import jax.numpy as jnp


def get_induced_velocity_doublet(
    doublet_point: vec3,
    query_point: vec3,
    doublet_strength: vec3,
    radius: float = 1e-8,
    dotted_with: vec3 | None = None,
) -> vec3 | float:
    """
    Get the velocity induced by a point doublet at a given query point.

    This function calculates the velocity field induced by a point doublet, with regularization to
    avoid singularities when evaluating points near the doublet. Note that a point doublet is equivalent
    to either a source-sink dipole, or a ring vortex shrunk to a point.

    For this case, if:
         - μ is the doublet strength + direction (i.e., a vector), and
         - r is the vector from the doublet point to the query point (i.e., a relative position vector),
    then the velocity induced by the doublet is given by:
         - V_i = μ / |r|³ - 3 (μ·r) * r / |r|⁵

    Args:
        doublet_point: The point where the doublet is located, with shape (3,).
        query_point: The point where the velocity is evaluated, with shape (3,).
        doublet_strength: The strength vector of the doublet, with shape (3,). The magnitude
                         represents the strength and the direction represents the orientation.
        radius: The characteristic radius for regularization. This parameter controls
                the smoothing of the velocity field near the doublet to prevent singularities.
                Larger values create more diffuse effects with lower peak velocities.
        dotted_with: Optional vector to dot the induced velocity with. If provided, returns a scalar instead
                     of a vector.

    Returns:
        If dotted_with is None, returns the induced velocity vector with shape (3,).
        Otherwise, returns the scalar dot product of the velocity with dotted_with.
    """
    # Compute the relative position vector
    # r: vec3 = doublet_point - query_point
    r: vec3 = query_point - doublet_point

    # Compute an approximation for 1 / |r|
    inv_norm_r: float = smoothed_inv(jnp.linalg.norm(r, axis=-1), radius=radius)

    # # Pre-compute quantities for better GPU performance
    # inv_norm_r_squared: float = inv_norm_r * inv_norm_r
    # inv_norm_r_cubed: float = inv_norm_r_squared * inv_norm_r

    # # Compute the induced_velocity directly with minimal operations
    # induced_velocity = (
    #     (doublet_strength - 3.0 * jnp.dot(doublet_strength, r) * r * inv_norm_r_squared)
    #     * inv_norm_r_cubed
    #     / (4.0 * jnp.pi)
    # )

    induced_velocity = (
        doublet_strength * inv_norm_r**3
        - 3.0 * jnp.dot(doublet_strength, r) * r * inv_norm_r**5
    ) / (4.0 * jnp.pi)

    if dotted_with is None:
        return induced_velocity
    else:
        return jnp.dot(induced_velocity, dotted_with)


if __name__ == "__main__":
    import pyvista as pv
    import equinox as eqx

    pv.set_jupyter_backend("client")

    doublet_point = jnp.array([0.0, 0.0, 0.0])
    doublet_strength = jnp.array([0.0, 0.0, 1.0])

    x = jnp.linspace(-1, 1, 50)
    y = jnp.linspace(-1, 1, 50)
    z = jnp.linspace(-1, 1, 50)
    X, Y, Z = np.meshgrid(x, y, z)
    field = pv.StructuredGrid(X, Y, Z)

    # Evaluate the velocity field at each point
    @eqx.filter_jit
    def get_velocity_at_point(query_point: vec3) -> vec3:
        return get_induced_velocity_doublet(
            doublet_point=doublet_point,
            query_point=query_point,
            doublet_strength=doublet_strength,
        )

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(center=doublet_point, radius=0.05), color="magenta")
    pl.add_mesh(
        pv.Arrow(start=doublet_point, direction=doublet_strength, scale=0.5),
        color="magenta",
    )

    field["velocity"] = jax.vmap(get_velocity_at_point)(field.points)

    plane = field.slice(origin=doublet_point, normal=doublet_strength)
    plane["normal_velocity"] = jnp.dot(
        plane["velocity"], doublet_strength / jnp.linalg.norm(doublet_strength)
    )
    pl.add_mesh(plane, scalars="normal_velocity", cmap="bwr", clim=(-10, 10))

    streamlines = field.streamlines(
        vectors="velocity",
        source_center=doublet_point,
        source_radius=1.0,
        n_points=500,
        max_time=10.0,
    )

    # Calculate velocity magnitude for each point in the streamlines
    velocity_magnitude = np.linalg.norm(streamlines["velocity"], axis=1)
    velocity_magnitude = np.clip(velocity_magnitude, 0.0, 10.0)
    log_velocity = np.log1p(velocity_magnitude)  # log1p to handle zero velocities
    opacity = log_velocity / np.max(log_velocity)  # Normalize to [0,1]
    streamlines["opacity"] = opacity

    pl.add_mesh(
        streamlines,
        cmap="turbo",
        line_width=2,
        render_lines_as_tubes=True,
        scalars="velocity",
        opacity="opacity",
    )

    pl.add_axes()
    pl.show_bounds()
    pl.show()
