# %%
import os
import sys
import time
from itertools import product
import sourcedefender

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.fftpack import idct
from torch.utils.data import TensorDataset, ConcatDataset, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchdiffeq import odeint

# Append project directories to system path
sys.path.extend(["../", "../../", "../../../", "./", "/"])

# Local imports
from lib.helper import *
from lib.batchJacobian import batchJacobian_PDE
from operators import DifferentialOperators
from PPT_solver import solve_poisson
# %%
# Note: Uncomment the following line if needed


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def GRF(alpha: float, tau: float, s: int, A: float, B: float) -> np.ndarray:
    """
    Generate a single Gaussian Random Field (GRF) using the Karhunen-Loève expansion.

    Args:
        alpha (float): Smoothness parameter controlling the field regularity.
        tau (float): Length scale parameter of the covariance operator.
        s (int): Grid size (s x s) for the spatial field.
        A (float): Lower bound for scaling the field.
        B (float): Upper bound for scaling the field.

    Returns:
        np.ndarray: Scaled GRF of shape (s, s) with values in [A, B].
    """
    # Generate random variables for KL expansion
    random_variables = np.random.normal(loc=0, scale=1, size=(s, s))

    # Create meshgrid for eigenvalue computation
    k1, k2 = np.meshgrid(range(s), range(s), indexing='ij')

    # Compute square root of eigenvalues for the covariance operator
    eigenvalues = tau ** (alpha - 1) * (np.pi**2 * (k1**2 + k2**2) + tau**2) ** (-alpha / 2)

    # Construct KL coefficients
    kl_coefficients = s * eigenvalues * random_variables
    kl_coefficients[0, 0] = 0  # Zero out the (0,0) component

    # Apply inverse discrete cosine transform to obtain the spatial field
    spatial_field = idct(idct(kl_coefficients, norm='ortho', axis=0), norm='ortho', axis=1)

    # Normalize to zero mean and unit variance
    normalized_field = (spatial_field - np.mean(spatial_field)) / np.std(spatial_field)

    # Scale to the interval [A, B]
    field_min, field_max = normalized_field.min(), normalized_field.max()
    scaled_field = A + (normalized_field - field_min) / (field_max - field_min) * (B - A)

    return scaled_field

def batch_grf(alpha: float, tau: float, s: int, A: float, B: float, batch_size: int) -> torch.Tensor:
    """
    Generate a batch of Gaussian Random Fields as a PyTorch tensor.

    Args:
        alpha (float): Smoothness parameter controlling the field regularity.
        tau (float): Length scale parameter of the covariance operator.
        s (int): Grid size (s x s) for each spatial field.
        A (float): Lower bound for scaling the fields.
        B (float): Upper bound for scaling the fields.
        batch_size (int): Number of GRF samples to generate.

    Returns:
        torch.Tensor: Batch of GRFs with shape (batch_size, s, s).
    """
    # Generate batch of GRF samples
    batch_data = [GRF(alpha, tau, s, A, B) for _ in range(batch_size)]

    # Convert to PyTorch tensor
    batch_tensor = torch.tensor(batch_data, dtype=torch.float32)

    return batch_tensor

# Define forcing function
def forcing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute a forcing term based on input coordinates x and y.

    The forcing term is defined as sin(2π((x*y) + x)) + cos(2π((x*y) + y)).

    Args:
        x (torch.Tensor): Input tensor representing x-coordinates.
        y (torch.Tensor): Input tensor representing y-coordinates.

    Returns:
        torch.Tensor: Forcing term computed as a combination of sine and cosine functions.
    """
    # Compute the forcing term: sin(2π((x*y) + x)) + cos(2π((x*y) + y))
    xy_product = x * y
    sine_term = torch.sin(2 * np.pi * (xy_product + x))
    cosine_term = torch.cos(2 * np.pi * (xy_product + y))
    return sine_term + cosine_term




def rhs(
    t: float,
    w_flat: torch.Tensor,
    operators: 'DifferentialOperators',
    F: torch.Tensor,
    nu: float,
    N: int,
    device: torch.device,
    dx: float,
    dy: float
) -> torch.Tensor:
    """
    Compute the right-hand side of a vorticity-based PDE using differential operators.

    This function calculates the time derivative of the vorticity field (dw/dt) for a PDE,
    typically used in an ODE solver. It involves solving a Poisson equation, computing gradients,
    and applying a forcing term.

    Args:
        t (float): Current time (unused in computation but required by ODE solver convention).
        w_flat (torch.Tensor): Flattened vorticity tensor of shape (batch_size, N*N).
        operators (DifferentialOperators): Instance of DifferentialOperators for computing gradients and Laplacian.
        F (torch.Tensor): Forcing term tensor, typically of shape (N, N).
        nu (float): Viscosity coefficient.
        N (int): Grid size (N x N) for the spatial field.
        device (torch.device): Device (CPU/GPU) for tensor computations.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.

    Returns:
        torch.Tensor: Flattened time derivative of vorticity (dw/dt) of shape (batch_size, N*N).
    """
    # Reshape flattened vorticity into [batch_size, N, N]
    w = w_flat.view(-1, N, N)

    # Solve Poisson equation to obtain stream function psi
    psi = solve_poisson(w, device, N, dx, dy)

    # Compute gradients of stream function and vorticity
    dpsi_dx, dpsi_dy = operators.grad(psi)
    dw_dx, dw_dy = operators.grad(w)

    # Compute Jacobian: dpsi_dy * dw_dx - dpsi_dx * dw_dy
    jacobian = dpsi_dy * dw_dx - dpsi_dx * dw_dy

    # Compute Laplacian of vorticity
    lap_w = operators.laplacian(w)

    # Expand forcing term to match batch dimensions
    F_expanded = F.unsqueeze(0).expand_as(w)

    # Compute right-hand side: nu * Laplacian(w) - Jacobian + F
    dwdt = nu * lap_w - jacobian + F_expanded

    # Flatten output for ODE solver compatibility
    return dwdt.view(-1, N * N)



def solver(
    w0_batch: torch.Tensor,
    t_span: torch.Tensor,
    operators: 'DifferentialOperators',
    F: torch.Tensor,
    nu: float,
    N: int,
    device: torch.device,
    dx: float,
    dy: float
) -> torch.Tensor:
    """
    Solve a vorticity-based PDE using the provided initial conditions and parameters.

    This function uses an ODE solver to compute the evolution of the vorticity field
    over a specified time span, leveraging the right-hand side function `rhs`.

    Args:
        w0_batch (torch.Tensor): Initial vorticity conditions, shape (batch_size, N, N).
        t_span (torch.Tensor): Array of time points for solving the PDE, shape (num_times,).
        operators (DifferentialOperators): Instance of DifferentialOperators for computing derivatives.
        F (torch.Tensor): Forcing term, shape (1, N, N) or (N, N).
        nu (float): Viscosity coefficient.
        N (int): Grid size (N x N) for the spatial field.
        device (torch.device): Device (CPU/GPU) for tensor computations.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.

    Returns:
        torch.Tensor: Solution of the PDE, shape (num_times, batch_size, N*N).
    """
    # Flatten initial conditions to shape (batch_size, N*N) for ODE solver
    w0_flat = w0_batch.view(-1, N * N)

    # Define the right-hand side function with extra arguments for odeint
    rhs_func = lambda t, w_flat: rhs(t, w_flat, operators, F, nu, N, device, dx, dy)

    # Solve the PDE using odeint
    w_flat_sol = odeint(
        func=rhs_func,
        y0=w0_flat,
        t=t_span,
        method='implicit_adams'
    )

    return w_flat_sol

def create_and_save_dataset(rank, world_size, initial_conditions, PATH, nu=0.001, dataset_segment_size=10, N=64, t0=0, t_end=3, steps=150):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Distribute initial conditions across GPUs
    n_Sample = initial_conditions.shape[0]
    samples_per_gpu = n_Sample // world_size
    extra_samples = n_Sample % world_size
    
    if rank < extra_samples:
        start_idx = rank * (samples_per_gpu + 1)
        end_idx = start_idx + samples_per_gpu + 1
    else:
        start_idx = rank * samples_per_gpu + extra_samples
        end_idx = start_idx + samples_per_gpu

    local_initial_conditions = initial_conditions[start_idx:end_idx].to(device)

    # Create time tensor
    t_tensor = torch.linspace(t0, t_end, steps, dtype=torch.float32, device=device)
    x = torch.linspace(0, 1, N, dtype=torch.float32, device=device)
    y = torch.linspace(0, 1, N, dtype=torch.float32, device=device)
    dx = x[1] - x[0]
    operators = DifferentialOperators(N, dx, device)
    
    X, Y = torch.meshgrid(x, y, indexing='ij')
    F = forcing(X, Y)  # Apply the forcing term

    batch_initial_conditions_list = []
    u_list = []
    jac_list = []

    for i in range(local_initial_conditions.shape[0] // dataset_segment_size):
        batch_initial_conditions = local_initial_conditions[i * dataset_segment_size:(i + 1) * dataset_segment_size]
        batch_initial_conditions.requires_grad_(True)
        batch_size = batch_initial_conditions.shape[0]
        
        print(f'GPU {rank}: Preparing {i}-th dataset segment...')
        u = solver(batch_initial_conditions, t_tensor, operators, F, nu, N, device, dx, dy=dx).permute(1, -1, 0).reshape(batch_size, N, N, steps)
        N_reduce = 2
        solution_u_last_time_step = u[:, ::N_reduce, ::N_reduce, -1]
        
        jac = batchJacobian_PDE(solution_u_last_time_step, batch_initial_conditions)
        jac_upscaled = upscale_tensor(jac=jac, N=N_reduce)

        batch_initial_conditions_list.append(batch_initial_conditions.detach())
        u_list.append(u[:, :, :, -1].detach())
        jac_list.append(jac_upscaled.detach())

    local_batch_initial_conditions = torch.cat(batch_initial_conditions_list, dim=0)
    local_u = torch.cat(u_list, dim=0)
    local_jac = torch.cat(jac_list, dim=0)        
    
    # Save local dataset
    local_dataset = TensorDataset(local_batch_initial_conditions, local_u, local_jac)
    torch.save(local_dataset, PATH + f'/NSE_Dataset_perturbed_DDP_rank_{rank}.pt')
    print(f"Dataset for rank {rank} saved at {PATH}/NSE_Dataset_perturbed_DDP_rank_{rank}.pt")

    # Synchronize all processes
    dist.barrier()
    cleanup()

def run_ddp(world_size, initial_conditions, PATH, nu=0.001, dataset_segment_size=10, N=64, t0=0, t_end=3, steps=150):
    torch.multiprocessing.spawn(
        create_and_save_dataset,
        args=(world_size, initial_conditions, PATH, nu, dataset_segment_size, N, t0, t_end, steps),
        nprocs=world_size,
        join=True
    )

def combine_datasets(world_size, PATH, case):
    print("Combining datasets from all GPUs...")
    all_initial_conditions = []
    all_u = []
    all_jac = []

    for i in range(world_size):
        local_dataset = torch.load(PATH + f'/NSE_Dataset_perturbed_DDP_rank_{i}.pt', map_location='cpu', weights_only=False)
        all_initial_conditions.append(local_dataset.tensors[0].cpu())
        all_u.append(local_dataset.tensors[1].cpu())
        all_jac.append(local_dataset.tensors[2].cpu())

    combined_initial_conditions = torch.cat(all_initial_conditions, dim=0)
    combined_u = torch.cat(all_u, dim=0)
    combined_jac = torch.cat(all_jac, dim=0)

    combined_dataset = TensorDataset(combined_initial_conditions, combined_u, combined_jac)
    torch.save(combined_dataset, PATH +  '/' + case + '_main_dataset.pt')
    print(f"Combined dataset saved at {PATH}/{case}_main_dataset.pt")

    for i in range(world_size):
        os.remove(PATH + f'/NSE_Dataset_perturbed_DDP_rank_{i}.pt')
    print("Individual GPU datasets removed.")

# Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    case = 'nse_multi_ic_J_ic'

    PATH_DATA = '/scratch/amb10399/DATA/NSE'
    ensure_directory(PATH_DATA)
    n_Sample = 10
    Ns = 50

    # Generate initial conditions once
    initial_conditions = batch_grf(alpha=3.0, tau=0.5, s=Ns, A=-2.5, B=2.5, batch_size=n_Sample).to(device)

    world_size = torch.cuda.device_count()
    run_ddp(world_size, initial_conditions, PATH=PATH_DATA, nu=0.001, dataset_segment_size=10, N=Ns, t0=0, t_end=3, steps=100)

    # Combine datasets after parallel processing
    combine_datasets(world_size, PATH_DATA, case)

    print('done')


