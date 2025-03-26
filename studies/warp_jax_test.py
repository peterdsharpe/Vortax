import warp as wp
import jax
import jax.numpy as jnp
from warp.jax_experimental import jax_kernel

wp.init()


@wp.kernel
def sum_of_squares(
    input: wp.array(dtype=float),
    output: wp.array(dtype=float),  # Length N  # Length 1
):
    tid = wp.tid()
    # Initialize the output to zero
    if tid == 0:
        output[0] = 0.0

    # wp.synchronize()

    # Compute square of current element
    square = input[tid] * input[tid]
    # wp.atomic_add(output, 0, square)
    # In this case, we could instead allocate to a length-N array and then sum - but I'm using this as a
    # stand-in example for a more complex kernel where the atomic add is harder to avoid.
    output[tid] = square


sum_of_squares_kernel_jax = jax_kernel(sum_of_squares, launch_dims=(10,))


# Function that wraps the Warp kernel for use with JAX
@jax.jit
def sum_of_squares_func_jax(x_jax: jnp.ndarray) -> jnp.ndarray:
    return sum_of_squares_kernel_jax(x_jax)[0]


if __name__ == "__main__":
    print("\n=== Basic Usage Demo ===")
    input = jnp.arange(10, dtype=jnp.float32)
    print(f"{input = }")
    true_result = jnp.sum(input**2, keepdims=True)
    print(f"{true_result = }")
    warp_result = sum_of_squares_func_jax(input)
    print(f"{warp_result = }")
    print(f"{jnp.allclose(warp_result, true_result) = }")

    print("\n=== Gradient Demo ===")
    print(f"{input = }")
    true_grad_x = 2.0 * input
    print(f"{true_grad_x = }")
    warp_grad_fn = jax.grad(lambda input: sum_of_squares_func_jax(input)[0])
    warp_grad_x = warp_grad_fn(input)
    print(f"{warp_grad_x = }")
    print(f"{jnp.allclose(warp_grad_x, true_grad_x) = }")
