import numpy as np
from jax.experimental import sparse
import jax.numpy as jnp
from jax.config import config
from jax import jit

config.update("jax_enable_x64", True)

def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix
        u: N x N grid state at timestep k
        epsilon: stability constant

    Returns:
        N x N Grid state at timestep k+1
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u

def get_A(N): 
    """
    Generates the matrix A used in the finite difference scheme of the 2D heat equation.
    
    Args:
        N (int): The dimension of the square grid.
        
    Returns:
        np.ndarray: A square matrix of size N^2 x N^2 representing the discretized Laplacian operator with Dirichlet boundary conditions.
    """
    # Total number of points
    n = N * N
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0
    A = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) + np.diag(diagonals[3], N) + np.diag(diagonals[4], -N)
    return A 
 
def get_sparse_A(N):
    """
    Generate a sparse matrix representation of the matrix A used for the heat equation.
    Args:
        N (int): The dimension of the grid (N x N).

    Returns:
        A_sp_matrix (BCOO): The sparse matrix representation of A in BCOO format.
    """
    n = N * N
    diagonals = [-4 * jnp.ones(n), jnp.ones(n-1), jnp.ones(n-1), jnp.ones(n-N), jnp.ones(n-N)]
    diagonals = [diagonals[0], diagonals[1].at[(N-1)::N].set(0), diagonals[2].at[(N-1)::N].set(0), diagonals[3], diagonals[4]]
    A = jnp.diag(diagonals[0]) + jnp.diag(diagonals[1], 1) + jnp.diag(diagonals[2], -1) + jnp.diag(diagonals[3], N) + jnp.diag(diagonals[4], -N)
    A_sp_matrix = sparse.BCOO.fromdense(A)
    return A_sp_matrix

def advance_time_numpy(u, epsilon):
    """
    Advances the simulation of the heat equation by one time step using NumPy's vectorized operations.

    Args:
        u (np.ndarray): The current state of the system, an N x N grid.
        epsilon (float): The stability constant.

    Returns:
        np.ndarray: The updated state of the system, an N x N grid.
    """
    N = u.shape[0]
    # pad to (N+2)x(N+2) for border conditions 
    u_padded = np.pad(u, pad_width=1, mode='constant', constant_values=0)

    u_next = u + epsilon * (
        np.roll(u_padded, shift=-1, axis=0)[1:-1, 1:-1] + # Up cuz shift for -1 is backwards and axis=0 gives row
        np.roll(u_padded, shift=1, axis=0)[1:-1, 1:-1] + # Down
        np.roll(u_padded , shift=-1, axis=1)[1:-1, 1:-1] + # Left cuz axix=1 is column
        np.roll(u_padded, shift=1, axis=1)[1:-1, 1:-1] - # Right
        4 * u # Center 
    )

    return u_next

@jit
def advance_time_jax(u, epsilon):
    """
    Advances the heat distribution on a grid by one time step using JAX for JIT-compiled execution.

    Args:
        u (jnp.ndarray): The current state of the grid as a 2D JAX array.
        epsilon (float): The diffusion coefficient.

    Returns:
        jnp.ndarray: The updated state of the grid as a 2D JAX array.
    """
    N = u.shape[0]
    u_padded = jnp.pad(u, pad_width=1, mode='constant', constant_values=0)
    u_next = u + epsilon * (
        jnp.roll(u_padded, shift=-1, axis=0)[1:-1, 1:-1] + # Up cuz shift for -1 is backwards and axis=0 gives row
        jnp.roll(u_padded, shift=1, axis=0)[1:-1, 1:-1] + # Down
        jnp.roll(u_padded , shift=-1, axis=1)[1:-1, 1:-1] + # Left cuz axix=1 is column
        jnp.roll(u_padded, shift=1, axis=1)[1:-1, 1:-1] - # Right
        4 * u # Center 
    )

    return u_next






