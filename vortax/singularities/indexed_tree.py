import itertools
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
from dataclasses import dataclass
from typing import Any, Optional
from functools import cached_property, partial
from vortax.types import vec3

import numpy as np


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("indices", "center", "size", "_all_points", "children"),
    meta_fields=("parent",),
)
@dataclass(frozen=True)
class IndexedTree:
    """
    A JAX-compatible N-dimensional spatial tree structure for efficient point cloud operations.

    This class implements a hierarchical spatial decomposition (i.e., an octree in 3D or
    quadtree in 2D) that recursively subdivides space into equal-sized hypercubes. Each node
    represents a region of space containing a subset of points from the original point cloud.

    The tree structure looks like this in 2D (quadtree), with the children contained in `children`:

    Root
    ├── (--) SW quadrant
    │   ├── (--) SW sub-quadrant
    │   ├── (-+) NW sub-quadrant
    │   ├── (+-) SE sub-quadrant
    │   └── (++) NE sub-quadrant
    ├── (-+) NW quadrant
    │   └── ...
    ├── (+-) SE quadrant
    │   └── ...
    └── (++) NE quadrant
        └── ...

    In 3D, each node has up to 8 children (octree), and in N dimensions, each node has up to 2^N children.
    The boolean tuple key (e.g., (True, False, True) in 3D) indicates the position of the child relative to
    the center of the parent node along each dimension, where False means "lower than center" and
    True means "higher than center" in that dimension.

    Unlike a kd-tree, each node represents a hypercube with equal side lengths (aspect ratio of 1),
    which is beneficial for certain algorithms like Barnes-Hut that rely on well-balanced spatial
    decomposition for efficient force calculations in N-body problems.

    The `indices` attribute stores the indices of points contained in this node, allowing efficient
    reference back to the original point cloud without duplicating point data.

    Memory Efficiency:
    The points array is only stored on the root node and is never copied or sliced to children.
    All other nodes only store integer indices that reference into the root's point array.
    The root-level points array can be accessed from any child node through the .all_points
    property, which recursively traverses up the tree until it reaches the root node.
    This approach minimizes memory usage while maintaining efficient access to point data.

    Additionally, the index-based approach allows child nodes to reference other array-like data
    that corresponds to the points but is stored separately. For example, in an N-body gravity
    simulation, you can use the same indices to access mass values for each point without
    duplicating that data throughout the tree.

    Users should typically create instances using the `from_points` class method rather than
    directly instantiating this class. This method handles the recursive subdivision of space
    and construction of the entire tree structure. Recursion stops when the number of points
    in a node is 0 or 1.

    Example:
        ```python
        points = np.random.randn(100, 3)  # Create 100 random points in 3D
        points /= np.linalg.norm(points, axis=1, keepdims=True)  # Normalize to unit sphere
        tree = IndexedTree.from_points(points)  # Create the tree
        print(tree)  # Print the tree structure
        ```
    """

    indices: Int[Array, " n_points_total"]
    center: Float[Array, " n_dim"]
    size: float
    _all_points: Optional[Float[Array, " n_points_total n_dim"]] = None
    parent: Optional["IndexedTree"] = None
    children: Optional[dict[tuple[bool, ...], "IndexedTree"]] = None

    def __repr__(self) -> str:
        """Return a string representation of the tree structure with indentation.

        By default, limits the tree expansion for readability and speed.
        """
        try:
            return self.repr_recursive(max_depth=2)
        except RecursionError as e:
            return f"{self.__class__.__name__}({self.n_points} pts, size={self.size:.6g}, center={self.center})"

    def repr_recursive(
        self,
        prefix: str = "",
        max_depth: Optional[int] = None,
        current_depth: int = 0,
        hide_empty: bool = False,
    ) -> str:
        """Recursively build the string representation of the tree.

        Args:
            prefix: The prefix to use for the current line
            max_depth: Maximum depth to display in the tree. If None, shows all levels. (May be slow for large trees.)
            current_depth: Current depth in the recursion (used internally)
            hide_empty: If True, hide children with 0 points

        Returns:
            A string representation of this node and its children
        """
        # Base representation of this node
        result = f"{self.__class__.__name__}({self.n_points} pts, size={self.size:.6g}, center={self.center})"

        # If this is a leaf node or we've reached max depth, we're done
        if self.is_leaf or (max_depth is not None and current_depth >= max_depth):
            if (
                not self.is_leaf
                and max_depth is not None
                and current_depth >= max_depth
            ):
                result += " [...]"  # Indicate there are more levels not shown
            return result

        # Sort children by their keys for consistent output
        sorted_children = sorted(self.children.items())

        # Filter out empty children if requested
        if hide_empty:
            sorted_children = [
                (key, child) for key, child in sorted_children if child.n_points > 0
            ]

        # If all children were filtered out, return just this node
        if not sorted_children:
            return result

        # Start with the current node's representation
        lines = []
        lines.append(result)

        # Process each child
        for i, (key, child) in enumerate(sorted_children):
            is_last = i == len(sorted_children) - 1

            # Create the key representation (e.g., "+-+")
            key_str = "".join("+" if k else "-" for k in key)
            key_str = f"({key_str})"

            # Determine the connector and next prefix
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            # Get child representation and split into lines
            child_repr = child.repr_recursive(
                next_prefix, max_depth, current_depth + 1, hide_empty
            )
            child_lines = child_repr.split("\n")

            # Add first line of child with connector and key
            lines.append(f"{prefix}{connector}{key_str} {child_lines[0]}")

            # Add remaining lines of child with proper indentation
            if len(child_lines) > 1:
                lines.extend(child_lines[1:])

        return "\n".join(lines)

    @cached_property
    def points(self) -> Float[Array, "n_points_tree n_dim"]:
        return self.all_points[self.indices]

    @property
    def all_points(self) -> Float[Array, "n_points_tree n_dim"]:
        if not self.is_root:  # If this is a child node, get the points from the parent
            try:
                return self.parent.all_points
            except AttributeError:  # Parent is None, so we are the root
                raise ValueError("Root node has no points.")
        else:
            return self._all_points

    @cached_property
    def n_dim(self) -> int:
        if self._all_points is not None:
            return self._all_points.shape[1]
        elif not self.is_root:
            return self.parent.n_dim
        else:
            raise ValueError(
                "Root node has no points from which to infer dimensionality."
            )

    @cached_property
    def bounds_min(self) -> Float[Array, " n_dim"]:
        return self.center - self.size / 2

    @cached_property
    def bounds_max(self) -> Float[Array, " n_dim"]:
        return self.center + self.size / 2

    @cached_property
    def is_leaf(self) -> bool:
        return self.children is None

    @cached_property
    def is_root(self) -> bool:
        return self.parent is None

    @cached_property
    def depth(self) -> int:
        if self.is_root:
            return 0
        else:
            return self.parent.depth + 1

    @cached_property
    def n_points(self) -> int:
        return len(self.indices)

    def __getitem__(self, indices: tuple[bool, ...]) -> "IndexedTree":
        if self.is_leaf:
            raise IndexError("Cannot index a leaf node")
        else:
            try:
                return self.children[indices]
            except KeyError:
                raise IndexError(
                    f"Index {indices} not found in children keys {self.children.keys()}"
                )

    @classmethod
    def from_points(
        cls,
        all_points: Float[Array, "n_points_total n_dim"],
        _indices: None | Int[Array, " n_points_total"] = None,
        _center: None | Float[Array, " n_dim"] = None,
        _size: None | Float[Array, " n_dim"] = None,
        _parent: Optional["IndexedTree"] = None,
        validate: bool = False,
    ) -> "IndexedTree":
        """
        Create a node from a set of points by recursively subdividing space.

        Args:
            points: Array of point coordinates with shape (n_points_total, n_dim)
            max_points_per_leaf: Maximum number of points allowed in a leaf node
            indices: Indices of points in the original array (used in recursion)
            depth: Current recursion depth (used to prevent excessive recursion)
            max_depth: Maximum allowed recursion depth

        Returns:
            A new IndexedTree instance representing the spatial decomposition
        """
        # Forces a NumPy cast, since this is much faster in NumPy than in JAX due to branching logic
        all_points = np.asarray(all_points)

        # Determine if this is the base case (root node) or a recursive case (child node)
        is_root: bool = (
            _indices is None and _center is None and _size is None and _parent is None
        )

        if validate:
            if is_root:
                assert (
                    _indices is not None
                    and _center is not None
                    and _size is not None
                    and _parent is not None
                ), "Root node must have all inputs provided."
            else:
                assert (
                    _indices is None
                    and _center is None
                    and _size is None
                    and _parent is not None
                ), "Child node must have no inputs provided."

        if is_root:
            n_points_total = all_points.shape[0]
            _indices = np.arange(n_points_total)

            mins = np.min(all_points, axis=0)
            maxs = np.max(all_points, axis=0)
            _center = (mins + maxs) / 2
            axis_sizes = maxs - mins
            _size = np.max(axis_sizes)

            _parent = None

        ### Now, build the node
        node = cls(
            _all_points=None,  # Will be filled in below
            indices=_indices,
            center=_center,
            size=_size,
            parent=_parent,
            children=None,  # Will be filled in below
        )

        if is_root:  # Performs this separately (rather than ternary) for speed
            node.__dict__["_all_points"] = all_points

        ### Now, fill in the children, if needed
        if len(_indices) > 1:
            children: dict[tuple[bool, ...], "IndexedTree"] = {}

            # Precompute boolean arrays for each dimension as a 2D array
            # Shape: (n_points, n_dim)
            is_upper_half = node.all_points[_indices] >= _center

            # Generate all possible octants
            octants = np.array(
                list(itertools.product([False, True], repeat=node.n_dim))
            )  # Shape: (2^n_dim, n_dim)

            # Precompute child centers for all octants
            child_centers = _center + np.where(
                octants,
                _size / 4,
                -_size / 4,
            )

            # Pre-allocate the boolean mask array to avoid recreating it in the loop
            mask = np.ones(
                (len(_indices), len(octants)), dtype=bool
            )  # Shape: (n_points, 2^n_dim)

            # Vectorized computation of masks for all octants at once
            for dim in range(node.n_dim):
                # Vectorized approach: create a mask for the current dimension
                # Shape: (n_points, n_octants)
                dim_mask = np.zeros((len(_indices), len(octants)), dtype=bool)

                # For each octant, set True where the point's position matches the octant's requirement
                # Broadcasting: is_upper_half[:, dim][:, None] has shape (n_points, 1)
                # octants[:, dim][None, :] has shape (1, n_octants)
                dim_mask = is_upper_half[:, dim][:, None] == octants[:, dim][None, :]

                # Update the overall mask
                mask &= dim_mask

            # Create children for each octant
            for idx, octant_index in enumerate(octants):
                child_indices = _indices[mask[:, idx]]

                # Only create children for non-empty octants
                if len(child_indices) > 0:
                    children[tuple(map(bool, octant_index))] = cls.from_points(
                        all_points=None,
                        _indices=child_indices,
                        _center=child_centers[idx],
                        _size=_size / 2,
                        _parent=node,
                    )

            # node.children = children
            node.__dict__["children"] = children

        return node

    def draw_pyvista(
        self,
        draw_points: bool = True,
        draw_bounds: bool = True,
        depth_limit: int = None,
        plotter: "pv.Plotter" = None,
        show: bool = True,
    ) -> "pv.Plotter":
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        assert self.n_dim == 3, (
            "Only 3D trees are supported for visualization with PyVista"
        )

        # Generate a colormap with distinct colors
        import matplotlib

        def _get_node_color(depth: int) -> tuple:
            """Generate a unique color for a node based on its ID."""
            cmap = matplotlib.colormaps["Dark2"]
            # return cmap(node_id)[:3]  # RGB values only
            return cmap(depth)[:3]  # RGB values only

        # Count total nodes to determine color distribution
        def _count_nodes(node, depth=0, current_depth=0):
            if node.is_leaf or (
                depth_limit is not None and current_depth >= depth_limit
            ):
                return 1
            return 1 + sum(
                _count_nodes(child, depth, current_depth + 1)
                for child in node.children.values()
            )

        total_nodes = _count_nodes(self, depth_limit)

        # Recursively draw the tree
        def _draw_node(node, node_id=0, depth=0):
            if depth_limit is not None and depth > depth_limit:
                return node_id

            color = _get_node_color(depth)

            # Draw bounding box if requested
            if draw_bounds:
                half_size = (
                    node.size / 2 * 1.02
                )  # Grow by 2% to allow for overlaps to be seen
                bounds = [
                    node.center[0] - half_size,
                    node.center[0] + half_size,
                    node.center[1] - half_size,
                    node.center[1] + half_size,
                    node.center[2] - half_size,
                    node.center[2] + half_size,
                ]
                box = pv.Box(bounds)
                plotter.add_mesh(
                    box, style="wireframe", color=color, line_width=2, opacity=0.9
                )
                plotter.add_mesh(box, style="surface", color=color, opacity=0.02)

            # Draw points if this is a leaf node and drawing points is requested
            if draw_points and node.is_leaf and node.n_points > 0:
                points_data = node.points
                point_cloud = pv.PolyData(points_data)
                plotter.add_mesh(
                    point_cloud, color="k", point_size=10, render_points_as_spheres=True
                )

            # Increment node_id for the next node
            next_id = node_id + 1

            # Recursively draw children
            if not node.is_leaf:
                for child in node.children.values():
                    next_id = _draw_node(child, next_id, depth + 1)

            return next_id

        # Start drawing from the root
        _draw_node(self)

        if show:
            plotter.show()

        return plotter


if __name__ == "__main__":
    points = np.random.randn(100, 3)
    points /= np.linalg.norm(
        points, axis=1, keepdims=True
    )  # Projected onto unit sphere
    import time

    start_time = time.time()
    node = IndexedTree.from_points(points)
    end_time = time.time()
    print(f"Tree construction time: {end_time - start_time:.4f} seconds")

    node.draw_pyvista(show=True)

    leaves, treedef = jax.tree.flatten(
        node, is_leaf=lambda x: isinstance(x, IndexedTree) and x.is_leaf
    )
    # print(f"{leaves = }")
    # print(f"{treedef = }")
