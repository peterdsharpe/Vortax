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

mesh_pv = pv.Sphere(center=(0.5, 0, 0.5), radius=0.5, direction=(1, 0, 0))

# mesh_pv = pv.read(Path(__file__).parent.parent / "datasets" / "motorBike_acvd.obj")
mesh_pv = pv.read(
    Path(__file__).parent.parent / "datasets" / "drivaer_4_acvd.stl"
).translate((0, 0, 0.318))

# Convert to Fluidity mesh

mesh = Mesh.from_pyvista(mesh_pv.clean(tolerance=1e-6).extract_surface())

face_centers = mesh.face_centers
face_normals = mesh.face_normals
face_areas = mesh.face_areas

freestream = jnp.array([1.0, 0.0, 0.0])


@eqx.filter_jit
def lhs(vortex_strengths: Float[Array, "n_faces"]) -> Float[Array, "n_faces"]:
    @eqx.filter_jit
    def compute_induced_normal_velocity(i: int) -> float:
        return jnp.sum(
            get_induced_velocity_mesh_ring_vortices(
                mesh=mesh,
                query_point=face_centers[i],
                dotted_with=face_normals[i],
                vortex_strengths=vortex_strengths,
            )
        )

    # Apply the vectorized function to all face indices in chunks to limit memory usage
    induced_normal_velocities: Float[Array, "n_faces"] = jax.lax.map(
        f=compute_induced_normal_velocity, xs=jnp.arange(mesh.n_faces), batch_size=64
    )

    return induced_normal_velocities


def preconditioner() -> Float[Array, "n_faces"]:

    @eqx.filter_jit
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


precond = lineax.TaggedLinearOperator(
    lineax.DiagonalLinearOperator(
        preconditioner(),
    ),
    tags=[
        lineax.symmetric_tag,
        lineax.diagonal_tag,
        lineax.positive_semidefinite_tag,
    ],
)

rhs = jnp.einsum("j,ij->i", -freestream, face_normals)

print("Solving...")

linear_op = lineax.FunctionLinearOperator(
    fn=lhs,
    input_structure=jnp.empty((mesh.n_faces,)),
)

solution = eqx.filter_jit(lineax.linear_solve)(
    operator=linear_op,
    vector=rhs,
    # solver=lineax.GMRES(
    #     rtol=1e-4,
    #     atol=1e-4,
    #     restart=50,
    #     stagnation_iters=100,
    # ),
    # options=dict(
    #     preconditioner=precond,
    # ),
)
vortex_strengths = solution.value

print(f"Velocity MAE error: {np.mean(np.abs(lhs(vortex_strengths) - rhs))}")
print(f"Vortex strengths mean: {np.mean(vortex_strengths)}")

@eqx.filter_jit
def compute_velocity(query_point: vec3) -> vec3:
    components = get_induced_velocity_mesh_ring_vortices(
        mesh=mesh,
        query_point=query_point,
        vortex_strengths=vortex_strengths,
    )
    return jnp.sum(components, axis=0) + freestream


### Now, visualize the flowfield in 3D with PyVista
res = 50
x = np.linspace(-3, 4, res * 7)
y = np.linspace(-1, 1, 3)
z = np.linspace(0, 3, res * 3)
X, Y, Z = np.meshgrid(x, y, z)
field = pv.StructuredGrid(X, Y, Z)

field.point_data["velocity"] = jax.lax.map(
    f=compute_velocity, xs=field.points, batch_size=64
)
slice_y = field.slice(normal="y")
velocity_viz = dict(
    scalars="velocity",
    cmap="turbo",
    clim=np.array([0.9, 1.1]) * np.linalg.norm(freestream),
)
pl = pv.Plotter()
pl.add_mesh(
    mesh_pv,
    scalars=vortex_strengths,
    cmap="coolwarm",
    clim=np.percentile(vortex_strengths, (5, 95)),
)
pl.add_mesh(
    slice_y,
    **velocity_viz,
    opacity=0.7,
    show_edges=True,
)
# pl.add_mesh(
#     field.streamlines_from_source(
#         # source=pv.PointSet(boundary_sample),
#         source=pv.PointSet(slice_y.points[np.random.choice(np.arange(len(slice_y.points)), size=500, replace=False)]),
#         vectors="velocity",
#     ),
#     **velocity_viz,
#     render_lines_as_tubes=True,
# )
pl.add_axes()
pl.show_bounds()
pl.add_title("Flow Visualization")
pl.show()
