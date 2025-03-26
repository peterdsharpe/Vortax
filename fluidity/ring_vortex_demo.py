import numpy as np
from fluidity.mesh import Mesh
import pyvista as pv
import jax.numpy as jnp
from jaxtyping import Float, Array
from fluidity.types import vec3
import jax
import equinox as eqx
import lineax
from pathlib import Path
from fluidity.ring_vortex import (
    get_induced_velocity_mesh_ring_vortices,
    get_induced_velocity_ring_vortex,
)

pv.set_jupyter_backend("client")

freestream = jnp.array([1.0, 0.0, 0.0])

# mesh_pv = pv.Sphere(center=(0.5, 0, 0.5), radius=0.5, direction=(1, 0, 0))
mesh_pv = pv.read(Path(__file__).parent.parent / "datasets" / "motorbike_cleaned.stl")
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
# mesh_pv = mesh_pv.compute_normals(
#     auto_orient_normals=True,
#     flip_normals=True,
# )

# Convert to Fluidity mesh
mesh = Mesh(
    vertices=jnp.array(mesh_pv.points),
    faces=jnp.array(mesh_pv.regular_faces),
)
# # Calculate dot product between freestream and face normals
# # Identify faces where flow is attached (arccos of dot product < 120 degrees)
# attached_mask = jnp.arccos(jnp.clip(jnp.einsum("j,ij->i", freestream, mesh.face_normals), -1.0, 1.0)) < jnp.radians(115)

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
# vertex_normals = jnp.array(mesh_pv.point_data["Normals"])


@eqx.filter_jit
def compute_self_induced_velocities() -> Float[Array, "n_faces"]:
    def compute_self_induced_velocity(i: int) -> float:
        return 1 / get_induced_velocity_ring_vortex(
            ring_points=mesh.vertices[mesh.faces[i]],
            query_point=face_centers[i],
            dotted_with=face_normals[i],
        )

    self_induced_velocities: Float[Array, "n_faces"] = jax.vmap(
        compute_self_induced_velocity
    )(jnp.arange(mesh.n_faces))
    return self_induced_velocities


self_induced_velocities = compute_self_induced_velocities()


@eqx.filter_jit
def lhs(vortex_strengths: Float[Array, "n_faces"]) -> Float[Array, "n_faces"]:
    def compute_face_induced_normal_velocity(i: int) -> float:
        return jnp.sum(
            get_induced_velocity_mesh_ring_vortices(
                mesh=mesh,
                query_point=face_centers[i],
                dotted_with=face_normals[i],
                vortex_strengths=vortex_strengths * self_induced_velocities,
                radius=1e-2 * smallest_face_scale,
            )
        )

    face_induced_normal_velocities: Float[Array, "n_faces"] = jax.lax.map(
        f=compute_face_induced_normal_velocity,
        xs=jnp.arange(mesh.n_faces),
        batch_size=64,  # Limits memory usage
    )

    # def compute_vertex_induced_normal_velocity(i: int) -> float:
    #     return jnp.sum(
    #         get_induced_velocity_mesh_ring_vortices(
    #             mesh=mesh,
    #             query_point=mesh.vertices[i],
    #             dotted_with=vertex_normals[i],
    #             vortex_strengths=vortex_strengths * self_induced_velocities,
    #             radius=1e-2 * smallest_face_scale,
    #         )
    #     )
    # vertex_induced_normal_velocities: Float[Array, "n_vertices"] = jax.lax.map(
    #     f=compute_vertex_induced_normal_velocity, xs=jnp.arange(mesh.n_vertices), batch_size=64
    # )

    return jnp.concatenate(
        [
            face_induced_normal_velocities,
            # vertex_induced_normal_velocities,
            # jnp.sum(vortex_strengths, keepdims=True)
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
#         y
#         ]),
#     y0=jnp.zeros(mesh.n_faces),
#     solver=optimistix.BestSoFarLeastSquares(optimistix.GaussNewton(
#         rtol=0.1, atol=0.1,
#         verbose=frozenset({"loss", "step_size"})
#     )),
# )

linear_op = lineax.FunctionLinearOperator(
    fn=lhs,
    input_structure=jnp.empty(mesh.n_faces),
)
solution = eqx.filter_jit(lineax.linear_solve)(
    operator=linear_op,
    vector=rhs,
    # solver=lineax.AutoLinearSolver(well_posed=False)
    # solver=lineax.SVD(),
    # solver=lineax.GMRES(
    #     rtol=1e-4,
    #     atol=1e-4,
    # ),
)

vortex_strengths = solution.value

print(f"Velocity MAE error: {np.mean(np.abs(lhs(vortex_strengths) - rhs))}")
print(f"Vortex strengths mean: {np.mean(vortex_strengths)}")
# print(f"Condition number: {eqx.filter_jit(jnp.linalg.cond)(linear_op.as_matrix()):.2e}")


@eqx.filter_jit
def compute_velocity(query_point: vec3) -> vec3:
    components = get_induced_velocity_mesh_ring_vortices(
        mesh=mesh,
        query_point=query_point,
        vortex_strengths=vortex_strengths * self_induced_velocities,
        radius=1e-1 * smallest_face_scale,
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
intensity = np.maximum(0, np.log10(np.abs(vortex_strengths) + 1e-10))
min_score, max_score = np.min(intensity), np.max(intensity)
opacities = (intensity - min_score) / (max_score - min_score)

pl_mesh = pv.Plotter()
pl_mesh.add_mesh(
    mesh.to_pyvista(),
    scalars=vortex_strengths, label="Vortex strengths",
    cmap="RdBu",
    clim=np.array([-1, 1]) * np.max(np.abs(np.percentile(vortex_strengths, (5, 95)))),
    opacity=opacities,
)
pl_mesh.add_axes()
pl_mesh.show_bounds()
pl_mesh.show()

# Visualize the distribution of vortex strengths
from fluidity.postprocessing.plot_symlog import plot_symlog_distribution

plot_symlog_distribution(vortex_strengths, show=True)
