import itertools
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
from dataclasses import dataclass
from typing import Any, Optional
from functools import cached_property, partial
from fluidity.types import vec3

import numpy as np

"""
Write this class, which is a JAX-compatible representation of an Octree node structure for a point cloud, generalized to n-dimensions (typically 3, but not necessarily). You should:
- Register it as a dataclass
- Form it with the from_points method, where you provide the point cloud, which is recursively split into a node.
- Keep both compute and memory complexity in at most N log N time. (For the most part, this should naturally happen, but don't do anything dumb)
- Each node leaf should store the indices in the original point cloud corresponding to the points in that leaf.

The end application will be to use this in a Barnes-Hut algorithm code for computing forces in a N-body problem. I will want to do something a little non-standard and advanced, which is that because the forces are computed at the points themselves, I want to do the hierarchical decomposition not just on the source points, but also on the target points (i.e., if two groups of points are far apart, not only should I lump the sources, but I should also lump the targets and apply the same force to all points in that target lump). 

You can assume that, unlike a typical N-body graviational simulation that is integrated in time, I'm only interested in computing the forces at a single snapshot in time. This means that the node will only be constructed once - there is no need to implement complicated logic to re-use past trees as a starting point for next-timestep node constructions, since the points do not move in this particular application.

Your algorithm should be thoughtfully constructed to support my needs. 
"""


@partial(jax.tree_util.register_dataclass)
@dataclass
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
        return self.repr_recursive(max_depth=3, hide_empty=True)
    
    def repr_recursive(self, prefix: str = "", max_depth: Optional[int] = None, current_depth: int = 0, hide_empty: bool = False) -> str:
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
            if not self.is_leaf and max_depth is not None and current_depth >= max_depth:
                result += " [...]"  # Indicate there are more levels not shown
            return result
        
        # Sort children by their keys for consistent output
        sorted_children = sorted(self.children.items())
        
        # Filter out empty children if requested
        if hide_empty:
            sorted_children = [(key, child) for key, child in sorted_children if child.n_points > 0]
        
        # If all children were filtered out, return just this node
        if not sorted_children:
            return result
        
        # Start with the current node's representation
        lines = []
        lines.append(result)
        
        # Process each child
        for i, (key, child) in enumerate(sorted_children):
            is_last = (i == len(sorted_children) - 1)
            
            # Create the key representation (e.g., "+-+")
            key_str = "".join("+" if k else "-" for k in key)
            key_str = f"({key_str})"
            
            # Determine the connector and next prefix
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            # Get child representation and split into lines
            child_repr = child.repr_recursive(next_prefix, max_depth, current_depth + 1, hide_empty)
            child_lines = child_repr.split("\n")
            
            # Add first line of child with connector and key
            lines.append(f"{prefix}{connector}{key_str} {child_lines[0]}")
            
            # Add remaining lines of child with proper indentation
            if len(child_lines) > 1:
                lines.extend(child_lines[1:])
        
        return "\n".join(lines)

    @property
    def points(self) -> Float[Array, "n_points_tree n_dim"]:
        return self.all_points()[self.indices]

    @property
    def all_points(self) -> Float[Array, "n_points_tree n_dim"]:
        if not self.is_root:  # If this is a child node, get the points from the parent
            try:
                return self.parent.all_points
            except AttributeError:  # Parent is None, so we are the root
                raise ValueError("Root node has no points.")
        else:
            return self._all_points

    @property
    def n_dim(self) -> int:
        if self._all_points is not None:
            return self._all_points.shape[1]
        elif not self.is_root:
            return self.parent.n_dim
        else:
            raise ValueError(
                "Root node has no points from which to infer dimensionality."
            )

    @property
    def bounds_min(self) -> Float[Array, " n_dim"]:
        return self.center - self.size / 2

    @property
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

    def __getitem__(self, *indices: tuple[bool, ...]) -> "IndexedTree":
        if self.is_leaf:
            raise IndexError("Cannot index a leaf node")
        else:
            return self.children[indices]

    @classmethod
    def from_points(
        cls,
        all_points: Float[Array, "n_points_total n_dim"],
        _indices: None | Int[Array, " n_points_total"] = None,
        _center: None | Float[Array, " n_dim"] = None,
        _size: None | Float[Array, " n_dim"] = None,
        _parent: Optional["IndexedTree"] = None,
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
        # all_points = np.asarray(all_points)

        # Determine if this is the base case (root node) or a recursive case (child node)
        inputs_required_for_children = [_indices, _center, _size, _parent]
        if all([i is not None for i in inputs_required_for_children]):
            is_child = True
        elif not any(inputs_required_for_children):
            is_child = False
        else:
            raise ValueError(
                f"Invalid inputs.\nFor children, all of {inputs_required_for_children} must be provided together.\nFor root, none of them should be provided.\nGot {inputs_required_for_children=}."
            )

        if not is_child:
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
            _all_points=None if is_child else all_points,
            indices=_indices,
            center=_center,
            size=_size,
            parent=_parent,
            children=None,
        )
        ### Now, fill in the children, if needed
        if node.n_points > 1:
            node_points = node.all_points[_indices]
            n_dim = node_points.shape[1]
            children: dict[tuple[bool, ...], "IndexedTree"] = {}
            
            # Precompute boolean arrays for each dimension as a 2D array
            # Shape: (n_points, n_dim)
            is_upper_half = node_points >= node.center
            
            # Generate all possible octants
            octants = list(itertools.product([False, True], repeat=n_dim))
            
            # Precompute child centers for all octants
            directions = np.array([[1.0 if use_upper else -1.0 for use_upper in octant] 
                                  for octant in octants])
            child_centers = node.center + _size / 4 * directions
            
            # Pre-allocate the boolean mask array to avoid recreating it in the loop
            mask = np.zeros((len(_indices), len(octants)), dtype=bool)
            mask[:] = True  # Start with all True
            
            # Vectorized computation of masks for all octants at once
            for dim in range(n_dim):
                for oct_idx, octant in enumerate(octants):
                    if octant[dim]:  # Upper half
                        mask[:, oct_idx] &= is_upper_half[:, dim]
                    else:  # Lower half
                        mask[:, oct_idx] &= ~is_upper_half[:, dim]
            
            # Create children for each octant
            for idx, octant_index in enumerate(octants):
                child_indices = _indices[mask[:, idx]]
                
                # Only create children for non-empty octants
                if len(child_indices) > 0:
                    children[octant_index] = cls.from_points(
                        all_points=None,
                        _indices=child_indices,
                        _center=child_centers[idx],
                        _size=_size / 2,
                        _parent=node,
                    )
        
            node.children = children

        return node


if __name__ == "__main__":
    points = np.random.randn(100000, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)  # Projected onto unit sphere
    node = IndexedTree.from_points(points)
    # node = eqx.filter_jit(IndexedTree.from_points)(points)
    # print(node)
    import pykdtree
    tree = pykdtree.kdtree.KDTree(points)
