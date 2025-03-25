from pathlib import Path

import pyvista as pv
import numpy as np
import re


def basic_cleaning(mesh: pv.PolyData, tolerance: float = 1e-6) -> pv.PolyData:
    # Removes duplicate vertices
    mesh = mesh.clean(tolerance=tolerance)

    # Removes duplicate faces
    if mesh.n_cells > 0:
        mesh = pv.PolyData.from_regular_faces(
            mesh.points, np.unique(np.sort(mesh.regular_faces, axis=1), axis=0)
        )
        mesh = mesh.compute_normals(
            point_normals=False,
            auto_orient_normals=True,
            flip_normals=True,  # Point normals outwards
        )
    return mesh


def process_mesh(mesh: pv.PolyData, n_verts: int = 5000) -> pv.PolyData:
    mesh = basic_cleaning(mesh)

    # import pymeshfix  # This is GPL-licensed, so we won't actually use it, but you could do something like this:
    # tin = pymeshfix.PyTMesh()
    # tin.load_array(mesh.points, mesh.regular_faces)
    # tin.remove_smallest_components()
    # tin.join_closest_components()
    # if tin.boundaries():
    #     tin.fill_small_boundaries()
    # tin.clean()
    # v, f = tin.return_arrays()
    # mesh = pv.PolyData.from_regular_faces(v, f)

    import pyacvd

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(1)
    clus.cluster(n_verts)
    mesh = clus.create_mesh()

    mesh = basic_cleaning(mesh)
    return mesh


def process_mesh_file(
    mesh_path: Path,
    output_path: Path = None,
    n_verts: int = 5000,
):
    if output_path is None:
        output_path = mesh_path.with_stem(f"{mesh_path.stem}_cleaned").with_suffix(
            ".stl"
        )

    mesh = process_mesh(mesh_path, n_verts)
    mesh.save(output_path)
    return mesh


def _get_one_groupid(mesh: pv.PolyData, idx: int) -> pv.PolyData:
    return mesh.threshold(
        int(idx) + np.array([-0.5, 0.5]), scalars="GroupIds"
    ).extract_surface()


def get_groupid(mesh: pv.PolyData, idx: int | slice) -> pv.PolyData:
    if isinstance(idx, int):
        return _get_one_groupid(mesh, idx)
    elif isinstance(idx, slice):
        return pv.merge(
            [
                _get_one_groupid(mesh, i)
                for i in range(idx.start, idx.stop, idx.step or 1)
            ]
        )
    else:
        raise TypeError(f"Expected int or slice, got {type(idx)}")


if __name__ == "__main__":
    pwd = Path(__file__).parent

    # mesh = process_mesh_file(pwd / "drivaer_4.stl", n_verts=5000)
    # print(mesh)
    # mesh.plot(color="w", show_edges=True)

    # mesh = pv.read(pwd / "motorbike.obj")
    # mesh = process_mesh(mesh)
    # mesh.save(pwd / "motorbike_cleaned.stl")

    mesh = pv.read(pwd / "motorbike.obj")
    # Extract region names from the OBJ file header
    region_names = []
    with open(pwd / "motorbike.obj", "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                # Use regex to extract the region name
                match = re.match(r"#\s+\d+\s+(.*?)%\d+", line)
                if match:
                    region_names.append(match.group(1))
            else:
                break

    raw_submeshes = [
        get_groupid(mesh, i) for i in range(int(np.max(mesh["GroupIds"])) + 1)
    ]
    meshes = {}
    for name, submesh in zip(region_names, raw_submeshes):
        if "shadow" in name.lower():
            print(f"Skipping shadow face `{name}`")
            continue
        if submesh.n_cells == 0:
            print(f"Submesh `{name}` has no cells")
            continue
        # submesh = process_mesh(submesh, n_verts=submesh.n_points // 10)
        if submesh.n_cells == 0:
            print(f"Submesh `{name}` has no cells after processing")
            continue

        meshes[name] = submesh

    import trimesh

    meshes = {
        k: trimesh.Trimesh(submesh.points, submesh.regular_faces)
        for k, submesh in meshes.items()
    }
    for k, mesh in meshes.items():
        trimesh.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
    mesh = pv.merge(list(meshes.values()))
    # mesh.plot(scalars="GroupIds", cmap="gist_ncar", annotations={
    #     k: f"{k}" for k in range(0, len(meshes), 5)
    # })
    mesh = process_mesh(mesh, n_verts=8000)
    mesh.plot(color="w", show_edges=True)
    mesh.save(pwd / "motorbike_cleaned.stl")
