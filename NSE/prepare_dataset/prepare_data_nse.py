# %%
# Standard library imports
import sys
import time

# Third-party imports
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.fftpack import idct
from torch.utils.data import ConcatDataset, random_split, TensorDataset
from torchdiffeq import odeint
from tqdm import tqdm
from itertools import product
import sourcedefender

# System path modifications
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("/")

# Local/custom module imports
from lib.batchJacobian import batchJacobian_PDE
from lib.helper import *
from operators import DifferentialOperators
from PPT_solver import solve_poisson


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MHPI()
print(f'Using device: {device}')

#%%

def GRF(alpha, tau, s, A, B):
    # Random variables in KL expansion
    xi = np.random.normal(0, 1, (s, s))
    
    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(range(s), range(s), indexing='ij')
    coef = tau**(alpha - 1) * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha / 2)
    
    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0  # Setting the (1,1) component to zero
    
    # Perform the inverse discrete cosine transform to obtain the spatial field
    U = idct(idct(L, norm='ortho', axis=0), norm='ortho', axis=1)
    
    # Normalize the field to have zero mean and unit variance
    U_normalized = (U - np.mean(U)) / np.std(U)
    
    # Scale to [A, B]
    U_scaled = A + (U_normalized - U_normalized.min()) / (U_normalized.max() - U_normalized.min()) * (B - A)
    
    return U_scaled

def batch_grf(alpha, tau, s, A, B, batch_size):
    batch_data = []
    for _ in range(batch_size):
        grf_sample = GRF(alpha, tau, s, A, B)
        batch_data.append(grf_sample)
    
    # Convert list of numpy arrays to a single PyTorch tensor
    batch_tensor = torch.tensor(batch_data, dtype=torch.float32)
    
    
    return batch_tensor



# Define forcing function
def forcing(x, y):
    return 1 * (torch.sin(2 * np.pi * ((x * y) + x)) + torch.cos(2 * np.pi * ((x * y) + y)))



def rhs(t, w_flat, operators, F, nu, N, device, dx, dy):
    """
    Calculate the right-hand side of the differential equation using given differential operators.

    Parameters:
    - t: Current time (not used in computation here, but required by ODE solver convention).
    - w_flat: Flattened tensor of current state variables.
    - operators: An instance of the DifferentialOperators class to compute derivatives and Laplacian.
    - F: Forcing term tensor.
    - nu: Viscosity coefficient.
    - N: Dimension size of the square grid (N x N).

    Returns:
    - Flattened tensor of derivatives of state variables.
    """
    # Reshape the flattened input into [batch_size, N, N]
    w = w_flat.view(-1, N, N)

    # Solve for psi using the Poisson equation solver
    psi = solve_poisson(w, device, N, dx, dy)  # This function needs to handle batch dimensions correctly

    # Using the provided DifferentialOperators class to compute derivatives
    dpsi_dx, dpsi_dy = operators.grad(psi)
    dw_dx, dw_dy = operators.grad(w)

    # Compute the Jacobian of the flow field
    jacobian = dpsi_dy * dw_dx - dpsi_dx * dw_dy

    # Compute the Laplacian of the vorticity field
    lap_w = operators.laplacian(w)

    # Compute the right-hand side of the PDE
    F_expanded = F.unsqueeze(0).expand_as(w)
    dwdt = nu * lap_w - jacobian + F_expanded

    # Flatten the output for compatibility with the ODE solver
    return dwdt.view(-1, N * N)


def solver(w0_batch, t_span, operators, F, nu, N, device, dx, dy):
    """
    Solves the PDE using provided initial conditions and parameters.
    
    Parameters:
        w0_batch (torch.Tensor): Initial conditions, shape (batch_size, N, N)
        t_span (torch.Tensor): Array of time points for which to solve the PDE
        operators (DifferentialOperators): Pre-computed differential operators
        F (torch.Tensor): Forcing term, shape (1, N, N)
        nu (float): Viscosity coefficient
        N (int): The dimension of the grid (NxN)
    
    Returns:
        torch.Tensor: The solution of the PDE reshaped to (batch_size, N, N, len(t_span))
    """
    # Flatten the initial conditions to match the input shape expected by odeint
    w0_flat = w0_batch.view(-1, N*N)

    # Solve the PDE using odeint with a lambda function to pass extra arguments to rhs
    w_flat_sol = odeint(
        lambda t, w_flat: rhs(t, w_flat, operators, F, nu, N, device, dx, dy),
        w0_flat,
        t_span,
        method='implicit_adams'
    )
    return w_flat_sol


def create_and_save_dataset(device, initial_conditions, nu=0.001, dataset_segment_size=10, N=64, t0=0, t_end=3, steps=150):
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
    main_dataset = None  # Initialize main_dataset as None
    for i in range(len(initial_conditions) // dataset_segment_size):

        t1 = time.time()

        batch_initial_conditions = initial_conditions[i * dataset_segment_size:(i + 1) * dataset_segment_size].to(device)
        batch_initial_conditions.requires_grad_(True)
        batch_size = batch_initial_conditions.shape[0]
        
        
        u = solver(batch_initial_conditions, t_tensor, operators, F, nu, N, device, dx, dy=dx).permute(1, -1, 0).reshape(batch_size, N, N, steps)
        N_reduce = 2
        solution_u_last_time_step = u[:, ::N_reduce, ::N_reduce, -1]
        
        jac = batchJacobian_PDE(solution_u_last_time_step, batch_initial_conditions)
        jac_upscaled = upscale_tensor(jac=jac, N=N_reduce)

        batch_size = u.shape[0]
        batch_initial_conditions_list.append(batch_initial_conditions.detach())
        u_list.append(u[:, :, :, -1].detach())
        jac_list.append(jac_upscaled.detach())

        print(f'Preparing {i}-th dataset segment, {time.time() - t1:.3f} s')

    batch_initial_conditions_temp = torch.cat(batch_initial_conditions_list, dim=0)
    u_temp = torch.cat(u_list, dim=0)
    jac_temp = torch.cat(jac_list, dim=0)        
    temp_dataset = TensorDataset(batch_initial_conditions_temp, u_temp, jac_temp)

    return temp_dataset

case = 'nse_multi_ic_J_ic'
PATH_SAVE_DATASET = '/scratch/amb10399/DATA/NSE'
ensure_directory(PATH_SAVE_DATASET)

n_Sample = 10
Ns = 50
initial_conditions =  batch_grf(alpha=3.0, tau=0.5, s=Ns, A=-1.0, B=1.0, batch_size=n_Sample).to(device)

t_start = time.time()
dataset = create_and_save_dataset(device, initial_conditions, nu=0.001, dataset_segment_size=10, N=Ns, t0=0, t_end=3, steps=100)
torch.save(dataset, PATH_SAVE_DATASET + '/' + case + '_main_dataset.pt')
print(f' --------  overal cpu time: {time.time() - t_start:.3f} s -------')
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulated data for demonstration: replace with your actual 3D data
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
x, y = np.meshgrid(x, y)
# z = dataset.tensors[1][9, :, :].cpu().numpy()
z = 1000 * dataset.tensors[-1][2, 25, 30, :, :].cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')  # Choose a colormap

plt.show()

# %%
