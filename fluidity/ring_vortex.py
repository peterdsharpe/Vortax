import jax
from fluidity.mesh import Mesh
from jaxtyping import Float, Array, Int
from fluidity.singularities.utils import smoothed_inv
from fluidity.types import vec3
import jax.numpy as jnp


def get_induced_velocity_ring_vortex(
    ring_points: Float[vec3, " n_ring_points"],
    query_point: vec3,
    vortex_strength: float = 1.0,
    radius: float = 1e-8,
    dotted_with: vec3 | None = None,
) -> vec3 | float:
    """
    Get the velocity induced by a single ring vortex at a given point.

    This function calculates the velocity field induced by a ring vortex using the Biot-Savart law
    with Kaufmann's regularization to avoid singularities when evaluating points near the vortex.

    Args:
        ring_points: The points defining the ring vortex, with shape (n_ring_points, 3).
        query_point: The point where the velocity is evaluated, with shape (3,).
        vortex_strength: The strength (circulation) of the vortex. Positive values correspond
                         to right-hand rule circulation.
        radius: The characteristic radius of the Kaufmann vortex core. This parameter controls
                the smoothing of the velocity field near the vortex filament to prevent singularities.
                Larger values create more diffuse vortices with lower peak velocities.
        dotted_with: Optional vector to dot the result with. If provided, returns a scalar instead
                     of a vector.

    Returns:
        If dotted_with is None, returns the induced velocity vector with shape (3,).
        Otherwise, returns the scalar dot product of the velocity with dotted_with.
    """
    a: Float[vec3, " n_ring_points"] = ring_points - query_point.reshape(1, -1)
    norm_a: Float[Array, " n_ring_points"] = jnp.linalg.norm(a, axis=-1)
    norm_a_inv: Float[Array, " n_ring_points"] = smoothed_inv(norm_a, radius)
    b: Float[vec3, " n_ring_points"] = jnp.roll(a, -1, axis=0)
    norm_b: Float[Array, " n_ring_points"] = jnp.roll(norm_a, -1)
    norm_b_inv: Float[Array, " n_ring_points"] = jnp.roll(norm_a_inv, -1)
    a_dot_b: Float[Array, " n_ring_points"] = jnp.sum(a * b, axis=-1)
    a_cross_b: Float[vec3, " n_ring_points"] = jnp.cross(a, b)

    contributions: Float[vec3, " n_ring_points"] = (
        a_cross_b
        * (norm_a_inv + norm_b_inv).reshape(-1, 1)
        * smoothed_inv(norm_a * norm_b + a_dot_b, radius).reshape(-1, 1)
    )
    if dotted_with is None:
        result = jnp.einsum("ij->j", contributions)
    else:
        result = jnp.einsum("ij,j->", contributions, dotted_with)

    return result * vortex_strength / (4.0 * jnp.pi)


def get_induced_velocity_mesh_ring_vortices(
    mesh: Mesh,
    query_point: vec3,
    vortex_strengths: Float[Array, " n_faces"] | None = None,
    radius: float = 1e-8,
    dotted_with: vec3 | None = None,
) -> Float[vec3, " n_faces"] | Float[Array, " n_faces"]:
    """
    Get the velocity induced by ring vortices on a mesh at a given point.

    This function calculates the velocity field induced by ring vortices placed on each face of the mesh
    using the Biot-Savart law with Kaufmann's regularization to avoid singularities. By computing the
    induced velocity from the whole mesh at once, rather than iterating over individual ring vortices,
    this implementation achieves better computational efficiency by avoiding repeated calculations. Note that this
    does increase the memory usage over vmapping over individual ring vortices, due to these stored precomputations.

    Args:
        mesh: The mesh containing faces that define ring vortices, with vertices of shape (n_vertices, 3).
        query_point: The point where the velocity is evaluated, with shape (3,).
        vortex_strengths: The strength (circulation) of each vortex, with shape (n_faces,).
                          Positive values correspond to right-hand rule circulation.
                          If None, unit strength is assumed for all vortices.
        radius: The characteristic radius of the Kaufmann vortex core. This parameter controls
                the smoothing of the velocity field near the vortex filament to prevent singularities.
                Larger values create more diffuse vortices with lower peak velocities.
        dotted_with: Optional vector to dot the result with. If provided, returns a scalar for each face
                     instead of a vector.

    Returns:
        If dotted_with is None, returns the induced velocity vectors with shape (n_faces, 3).
        Otherwise, returns the scalar dot products of the velocities with dotted_with, shape (n_faces,).
    """
    # Set defaults
    if vortex_strengths is None:
        vortex_strengths = jnp.ones(mesh.n_faces)

    r: Float[vec3, " n_vertices"] = mesh.vertices - query_point.reshape(1, -1)
    norm_r: Float[Array, " n_vertices"] = jnp.linalg.norm(r, axis=-1)
    norm_r_inv: Float[Array, " n_vertices"] = smoothed_inv(norm_r, radius)

    # Get all vertex pairs for each face (3 pairs per face)
    face_indices: Int[Array, " n_edges"] = jnp.repeat(
        jnp.arange(mesh.n_faces), 3
    )  # points from each edge to its parent face
    a_indices: Int[Array, " n_edges"] = mesh.faces.reshape(-1)  # Flatten all vertices
    b_indices: Int[Array, " n_edges"] = jnp.roll(mesh.faces, -1, axis=1).reshape(
        -1
    )  # Shifted vertices for pairs

    # Get vectors from query point to vertices
    a: Float[vec3, " n_edges"] = r[a_indices]
    b: Float[vec3, " n_edges"] = r[b_indices]
    norm_a: Float[Array, " n_edges"] = norm_r[a_indices]
    norm_b: Float[Array, " n_edges"] = norm_r[b_indices]
    norm_a_inv: Float[Array, " n_edges"] = norm_r_inv[a_indices]
    norm_b_inv: Float[Array, " n_edges"] = norm_r_inv[b_indices]

    # Compute dot and cross products
    a_dot_b: Float[Array, " n_edges"] = jnp.sum(a * b, axis=1)
    a_cross_b: Float[vec3, " n_edges"] = jnp.cross(a, b)

    # Compute contribution for each edge
    contributions: Float[vec3, " n_edges"] = (
        a_cross_b
        * (norm_a_inv + norm_b_inv).reshape(-1, 1)
        * smoothed_inv(norm_a * norm_b + a_dot_b, radius).reshape(-1, 1)
    )

    # Sum contributions over all edges
    if dotted_with is None:
        result: Float[vec3, " n_faces"] = jax.ops.segment_sum(
            contributions, face_indices, mesh.n_faces
        )
    else:
        dotted_contributions: Float[Array, " n_edges"] = jnp.sum(
            contributions * dotted_with, axis=1
        )
        result: Float[Array, " n_faces"] = jax.ops.segment_sum(
            dotted_contributions, face_indices, mesh.n_faces
        )

    result = jnp.einsum("i...,i->i...", result, vortex_strengths / (4.0 * jnp.pi))

    return result
