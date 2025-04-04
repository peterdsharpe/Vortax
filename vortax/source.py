import jax
import numpy as np
from vortax.mesh import Mesh
from jaxtyping import Float, Array, Int
from vortax.singularities.utils import smoothed_inv
from vortax.types import vec3
import jax.numpy as jnp


def get_induced_velocity_source(
    source_point: vec3,
    query_point: vec3,
    source_strength: float,
    radius: float = 1e-8,
    dotted_with: vec3 | None = None,
) -> vec3 | float:
    """
    Get the velocity induced by a point source at a given query point.

    This function calculates the velocity field induced by a point source, with regularization to
    avoid singularities when evaluating points near the source.

    For this case, if:
         - σ is the source strength (a scalar), and
         - r is the vector from the source point to the query point (i.e., a relative position vector),
    then the velocity induced by the source is given by:
         - V = σ * r / |r|³ / (4π)

    Args:
        source_point: The point where the source is located, with shape (3,).
        query_point: The point where the velocity is evaluated, with shape (3,).
        source_strength: The strength of the source (scalar). Positive values represent
                        outflow (source), negative values represent inflow (sink).
        radius: The characteristic radius for regularization. This parameter controls
                the smoothing of the velocity field near the source to prevent singularities.
                Larger values create more diffuse effects with lower peak velocities.
        dotted_with: Optional vector to dot the induced velocity with. If provided, returns a scalar instead
                     of a vector.

    Returns:
        If dotted_with is None, returns the induced velocity vector with shape (3,).
        Otherwise, returns the scalar dot product of the velocity with dotted_with.
    """
    # Compute the relative position vector
    r: vec3 = query_point - source_point

    # Compute an approximation for 1 / |r|
    inv_norm_r: float = smoothed_inv(jnp.linalg.norm(r, axis=-1), radius=radius)

    # Compute the induced velocity
    induced_velocity = source_strength * r * inv_norm_r**3 / (4.0 * jnp.pi)

    if dotted_with is None:
        return induced_velocity
    else:
        return jnp.dot(induced_velocity, dotted_with)


if __name__ == "__main__":
    import pyvista as pv
    import equinox as eqx

    pv.set_jupyter_backend("client")

    source_point = jnp.array([0.0, 0.0, 0.0])
    source_strength = 1.0  # Positive for source, negative for sink

    x = jnp.linspace(-1, 1, 50)
    y = jnp.linspace(-1, 1, 50)
    z = jnp.linspace(-1, 1, 50)
    X, Y, Z = np.meshgrid(x, y, z)
    field = pv.StructuredGrid(X, Y, Z)

    # Evaluate the velocity field at each point
    @eqx.filter_jit
    def get_velocity_at_point(query_point: vec3) -> vec3:
        return get_induced_velocity_source(
            source_point=source_point,
            query_point=query_point,
            source_strength=source_strength,
        )

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(center=source_point, radius=0.05), color="red")

    field["velocity"] = jax.vmap(get_velocity_at_point)(field.points)

    # Calculate velocity magnitude for visualization
    field["velocity_magnitude"] = np.linalg.norm(field["velocity"], axis=1)

    # Create a slice plane
    plane = field.slice(origin=source_point, normal=[0, 0, 1])
    pl.add_mesh(plane, scalars="velocity_magnitude", cmap="turbo", clim=(0, 5))

    streamlines = field.streamlines(
        vectors="velocity",
        source_center=source_point,
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
        scalars="velocity_magnitude",
        opacity="opacity",
    )

    pl.add_axes()
    pl.show_bounds()
    pl.show()
