# :tornado: Vortax :tornado:

> NOTE: this library is a heavy work in progress - expect breaking changes.

Vortax is a small self-contained JAX-based 3D potential flow solver.

## Features

- **Hardware-accelerated**: Leverages JAX to run parts of the solve (e.g., kernel computations, linear solves) on GPUs.
- **Scalable**: Uses matrix-free methods and hierarchical spatial decompositions to improve runtime for large problems. (WIP)
- **Differentiable**: Differentiable with respect to a) geometry, b) freestream conditions, and c) singularity kernel functions.
- **Mesh-compatible**: Takes raw triangulated surface meshes (e.g., STL files) as input, allowing easy application to complex geometries.

## Gallery

[Interactive motorbike demo](https://peterdsharpe.github.io/Vortax/motorbike_scene.html):

[![Motorbike demo](./assets/motorbike.jpg)](https://peterdsharpe.github.io/Vortax/motorbike_scene.html)

Octree decomposition for hierarchical acceleration:

![Octree decomposition](./assets/octree.jpg)

## Installation

This is a platform-agnostic installation:

```
git clone git@github.com:peterdsharpe/Vortax.git
cd Vortax
pip install -e .
```

By default, this will pull the CPU-only version of `jax` and `jaxlib`, which is compatible with Linux/MacOS/WSL/Windows. To use GPU acceleration, you'll want to first [install `jax` and `jaxlib` with GPU support](https://docs.jax.dev/en/latest/installation.html). On Linux/MacOS/WSL, you can typically do this with the following, assuming you have CUDA 12 installed (`nvidia-smi` to check):

```
pip install -U "jax[cuda12]"
```
