import numpy as np
from vortax.mesh import Mesh
import pyvista as pv
import jax.numpy as jnp
from jaxtyping import Float, Array
from vortax.types import vec3
import jax
import equinox as eqx
import lineax
from vortax.doublet import get_induced_velocity_doublet
from vortax.postprocessing.plot_symlog import plot_symlog_distribution
from vortax.ring_vortex import get_induced_velocity_mesh_ring_vortices

pv.set_jupyter_backend("client")

freestream = jnp.array([1.0, 0.0, 0.0])

mesh_pv = pv.Sphere(center=(0.5, 0, 0.5), radius=0.5, direction=(1, 0, 0))
# mesh_pv = pv.read(Path(__file__).parent.parent / "datasets" / "motorbike_cleaned.stl")
# mesh_pv = pv.read(
#     Path(__file__).parent.parent / "datasets" / "drivaer_4_cleaned.stl"
# ).translate((0, 0, 0.318))

mesh_pv = mesh_pv.compute_cell_sizes(length=False, area=True, volume=False)
smallest_face_scale = mesh_pv.cell_data["Area"].min() ** 0.5
mesh_pv = mesh_pv.clean(tolerance=smallest_face_scale)
# mesh_pv = pv.PolyData.from_regular_faces(
#     mesh_pv.points, np.unique(np.sort(mesh_pv.regular_faces, axis=1), axis=0)
# )
# mesh_pv = mesh_pv.clean(tolerance=smallest_face_scale)
mesh_pv = mesh_pv.compute_normals(
    auto_orient_normals=True,
    flip_normals=True,
)

# Convert to vortax mesh
mesh = Mesh(
    vertices=jnp.array(mesh_pv.points),
    faces=jnp.array(mesh_pv.regular_faces),
)
# # Calculate dot product between freestream and face normals
# # Identify faces where flow is attached (arccos of dot product < 120 degrees)
# attached_mask = jnp.arccos(jnp.clip(jnp.einsum("j,ij->source_i", freestream, mesh.face_normals), -1.0, 1.0)) < jnp.radians(115)

# # Get indices of attached faces
# attached_faces_indices = jnp.where(attached_mask)[0]

# # Create new mesh with only attached faces
# mesh = Mesh(
#     vertices=mesh.vertices,
#     faces=mesh.faces[attached_faces_indices]
# )

face_centers = mesh.face_centers
face_normals = mesh.face_normals
face_areas = mesh.face_areas
vertex_normals = jnp.array(mesh_pv.point_data["Normals"])


@eqx.filter_jit
def lhs(doublet_strengths: Float[Array, " n_faces"]) -> Float[Array, " n_faces"]:
    def get_induced_velocity_single_pair(source_i: int, target_i: int) -> float:
        return get_induced_velocity_doublet(
            doublet_point=face_centers[source_i],
            # query_point=mesh.vertices[target_i],
            query_point=face_centers[target_i],
            doublet_strength=doublet_strengths[source_i]
            * face_normals[source_i]
            * face_areas[source_i] ** 2,
            # radius=smallest_face_scale,
            # dotted_with=vertex_normals[target_i],
            dotted_with=face_normals[target_i],
        )

    # First, vmap over source faces for each target face
    def get_induced_velocities_on_target(target_i: int) -> float:
        """
        Compute the sum of induced normal velocities from all source faces on a target face.

        Args:
            target_i: Index of the target face

        Returns:
            The total normal velocity induced at the target face
        """
        # Vmap over all source faces for this target face
        induced_velocities = jax.vmap(
            lambda source_i: get_induced_velocity_single_pair(source_i, target_i)
        )(jnp.arange(mesh.n_faces))

        # Sum all contributions
        return jnp.sum(induced_velocities, axis=-1)

    # Map over all target faces with batching for memory efficiency
    face_induced_normal_velocities = jax.lax.map(
        f=get_induced_velocities_on_target,
        xs=jnp.arange(mesh.n_faces),
        batch_size=64,  # Limits memory usage
    )

    return jnp.concatenate(
        [
            face_induced_normal_velocities,
            # vertex_induced_normal_velocities,
            # jnp.sum(doublet_strengths, keepdims=True)
        ]
    )


rhs = jnp.concatenate(
    [
        jnp.einsum("j,ij->i", -freestream, face_normals),
        # jnp.einsum("j,ij->i", -freestream, vertex_normals),
        # jnp.array([0.0])
    ]
)

print("Solving...")

# import optimistix
# solution = eqx.filter_jit(optimistix.least_squares)(
#     fn=lambda y, args: jnp.concatenate([
#         lhs(y) - rhs,
#         # y
#         ]),
#     y0=jnp.zeros(mesh.n_faces),
#     solver=optimistix.BestSoFarLeastSquares(optimistix.GaussNewton(
#         rtol=0.1, atol=0.1,
#         verbose=frozenset({"loss", "step_size"}),
#     )),
# )

linear_op = lineax.FunctionLinearOperator(
    fn=lhs,
    input_structure=jnp.empty(mesh.n_faces),
)
solution = eqx.filter_jit(lineax.linear_solve)(
    operator=linear_op,
    vector=rhs,
    solver=lineax.AutoLinearSolver(well_posed=False),
    # solver=lineax.SVD(),
    # solver=lineax.GMRES(
    #     rtol=1e-4,
    #     atol=1e-4,
    # ),
)

doublet_strengths = solution.value

print(f"Velocity MAE error: {np.mean(np.abs(lhs(doublet_strengths) - rhs))}")
print(f"Vortex strengths mean: {np.mean(doublet_strengths)}")
print(f"Condition number: {eqx.filter_jit(jnp.linalg.cond)(linear_op.as_matrix()):.2e}")


# @eqx.filter_jit
# def compute_velocity(query_point: vec3) -> vec3:
#     """
#     Compute the velocity at a query point due to all doublets on the mesh plus freestream.

#     Args:
#         query_point: The point where the velocity is evaluated, with shape (3,).

#     Returns:
#         The total velocity vector at the query point.
#     """
#     def get_velocity_from_face(source_i: int) -> vec3:
#         return get_induced_velocity_doublet(
#             doublet_point=face_centers[source_i],
#             query_point=query_point,
#             doublet_strength=doublet_strengths[source_i] * face_normals[source_i] * face_areas[source_i] ** 2,
#             # radius=1 * smallest_face_scale,
#         )

#     # Map over all faces with batching for memory efficiency
#     induced_velocities = jax.lax.map(
#         f=get_velocity_from_face,
#         xs=jnp.arange(mesh.n_faces),
#         batch_size=64,  # Limits memory usage
#     )

#     # Sum all contributions and add freestream
#     return jnp.sum(induced_velocities, axis=0) + freestream



@eqx.filter_jit
def compute_velocity(query_point: vec3) -> vec3:
    components = get_induced_velocity_mesh_ring_vortices(
        mesh=mesh,
        query_point=query_point,
        vortex_strengths=doublet_strengths * face_areas**0.5,
        radius=1e-2 * smallest_face_scale,
    )
    return jnp.sum(components, axis=0) + freestream


### Now, visualize the flowfield in 3D with PyVista
res = 150
x = np.linspace(-3, 6, res * 9)
y = np.linspace(-1, 1, 3)
z = np.linspace(0, 3, res * 3)
X, Y, Z = np.meshgrid(x, y, z)
field = pv.StructuredGrid(X, Y, Z)

field.point_data["velocity"] = jax.lax.map(
    f=compute_velocity, xs=field.points, batch_size=256
)
slice_y = field.slice(normal="y")
velocity_viz = dict(
    scalars="velocity",
    label="Velocity",
    cmap="turbo",
    clim=np.array([0.9, 1.1]) * np.linalg.norm(freestream),
)
pl = pv.Plotter()
pl.add_mesh(
    mesh.to_pyvista(),
    color="white",
    edge_color="lightgray",
    edge_opacity=0.5,
    show_edges=True,
)
pl.add_mesh(slice_y, **velocity_viz, opacity=0.3)
source = slice_y.points
# bounds = mesh.bounds
# mask = np.all(
#     np.logical_and(
#         source > bounds[:, 0],
#         source < bounds[:, 1],
#     ),
#     axis=1,
# )
# source = source[~mask]
source = source[np.random.choice(np.arange(len(source)), size=500, replace=False)]

pl.add_mesh(
    field.streamlines_from_source(
        source=pv.PointSet(source),
        vectors="velocity",
        compute_vorticity=False,
    ),
    **velocity_viz,
    render_lines_as_tubes=True,
)
pl.add_axes()
pl.show_bounds()
pl.show()

### Create a new plotter to visualize the mesh with vortex strengths
# intensity = np.maximum(0, np.log10(np.abs(doublet_strengths) + 1e-10))
# min_score, max_score = np.min(intensity), np.max(intensity)
# opacities = (intensity - min_score) / (max_score - min_score)

pl_mesh = pv.Plotter()
pl_mesh.add_mesh(
    mesh.to_pyvista(),
    scalars=doublet_strengths,
    label="Vortex strengths",
    cmap="RdBu",
    clim=np.array([-1, 1]) * np.max(np.abs(np.percentile(doublet_strengths, (5, 95)))),
    # opacity=opacities,
)
pl_mesh.add_axes()
pl_mesh.show_bounds()
pl_mesh.show()

# Visualize the distribution of vortex strengths

plot_symlog_distribution(doublet_strengths, show=True)
