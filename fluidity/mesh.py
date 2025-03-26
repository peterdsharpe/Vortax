from functools import lru_cache
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int, Array
from fluidity.singularities.utils import smoothed_inv
from fluidity.types import vec3
import pyvista as pv
import trimesh


class Mesh(eqx.Module):
    """
    This class represents a surface mesh.

    In general, this mesh:
    - Is all triangles
    - Is unstructured (e.g., no particular structure to the connectivity of the faces)
    - Does not need to be watertight, though physics are more theoretically-sound for closed surfaces
    - Should not be self-intersecting, nor have duplicate faces - this will cause linear algebra issues
    - May have duplicate vertices, though sharing vertices will speed up some computations

    The mesh is stored as two arrays:
    - vertices: a float[n_vertices, 3] array of vertex positions, where each row is a vertex
    - faces: a int[n_faces, 3] array of indices into the vertex array, representing the vertices of each face

    """

    vertices: Float[vec3, " n_vertices"]
    faces: Int[Array, "n_faces 3"]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def face_centers(self) -> Float[vec3, " n_faces"]:
        return jnp.mean(self.vertices[self.faces], axis=1)

    @property
    def face_normals(self) -> Float[vec3, " n_faces"]:
        cross = jnp.cross(
            self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
            self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 1]],
        )
        return cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)

    @property
    def vertex_normals(self) -> Float[vec3, "n_vertices"]:
        """
        Compute the normal vector at each vertex by averaging the normals of adjacent faces.

        This is a challenging operation because each vertex can be connected to a variable
        number of faces, creating a ragged array structure. To handle this in a JAX-compatible
        way, we:

        1. Create a mask indicating which faces each vertex belongs to
        2. Use this mask to compute a weighted sum of face normals for each vertex
        3. Normalize the resulting vectors to get unit normals

        Returns:
            Float[vec3, "n_vertices"]: Unit normal vectors for each vertex
        """
        # Create a mask of shape [n_vertices, n_faces] where mask[v, f] = 1.0 if vertex v is in face f
        vertex_indices: Int[Array, " n_vertices"] = jnp.arange(self.n_vertices)
        face_indices: Int[Array, " n_faces"] = jnp.arange(self.n_faces)

        # Reshape for broadcasting: [n_vertices, 1, 1] and [1, n_faces, 3]
        vertices_reshaped: Int[Array, "n_vertices 1 1"] = vertex_indices[:, None, None]
        faces_reshaped: Int[Array, "1 n_faces 3"] = self.faces[None, :, :]

        # Create mask: 1.0 where vertex is in face, 0.0 otherwise
        # Result shape: [n_vertices, n_faces, 3]
        mask: Float[Array, "n_vertices n_faces 3"] = (
            vertices_reshaped == faces_reshaped
        ).astype(jnp.float32)

        # Reduce along the last axis to get a mask of shape [n_vertices, n_faces]
        # If any position in the face matches the vertex, the result is 1.0
        mask: Float[Array, "n_vertices n_faces"] = jnp.any(mask, axis=-1).astype(
            jnp.float32
        )

        # Get face normals and areas for weighting
        face_normals: Float[vec3, " n_faces"] = self.face_normals
        face_areas: Float[Array, " n_faces"] = self.face_areas

        # Weight face normals by face areas for better quality normals
        weighted_normals: Float[vec3, " n_faces"] = face_normals * face_areas[:, None]

        # Use the mask to compute the sum of weighted face normals for each vertex
        # [n_vertices, n_faces] @ [n_faces, 3] -> [n_vertices, 3]
        summed_normals: Float[vec3, " n_vertices"] = mask @ weighted_normals

        # Normalize to get unit vectors
        # Add a small epsilon to avoid division by zero
        norms: Float[Array, "n_vertices 1"] = jnp.linalg.norm(
            summed_normals, axis=-1, keepdims=True
        )
        return summed_normals / jnp.maximum(norms, 1e-10)

    @property
    def face_areas(self) -> Float[Array, "n_faces"]:
        cross = jnp.cross(
            self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
            self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 1]],
        )
        return jnp.linalg.norm(cross, axis=-1) / 2

    @property
    def bounds(self) -> Float[Array, "3 2"]:
        mins = jnp.min(self.vertices, axis=0)
        maxs = jnp.max(self.vertices, axis=0)
        return jnp.stack([mins, maxs], axis=-1)

    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> "Mesh":
        return cls(
            vertices=jnp.array(mesh.points),
            faces=jnp.unique(mesh.regular_faces, axis=0),
        )

    def to_pyvista(self) -> pv.PolyData:
        return pv.PolyData.from_regular_faces(
            np.array(self.vertices), np.array(self.faces)
        )

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> "Mesh":
        return cls(vertices=jnp.array(mesh.vertices), faces=jnp.array(mesh.faces))


if __name__ == "__main__":
    import pyvista as pv

    mesh = pv.Sphere(
        # theta_resolution=10,
        # phi_resolution=10,
    )
    mesh = Mesh.from_pyvista(mesh)
    print(mesh)
    # mesh.to_pyvista().plot()
    self = mesh

    res = mesh.get_induced_velocity_components_ring_vortices(jnp.array([0.0, 0.0, 0.0]))
    print(res)
    res = mesh.get_induced_velocity_components_ring_vortices(
        jnp.array([0.0, 0.0, 0.0]), dotted_with=jnp.array([1.0, 0.0, 0.0])
    )
    print(res)
