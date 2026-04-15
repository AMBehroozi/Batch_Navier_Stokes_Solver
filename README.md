# Batch Navier-Stokes Solver

This repository contains a PyTorch-based, GPU-accelerated implementation of a Batch Navier-Stokes Equations (NSE) solver. It is designed to efficiently solve the 2D Navier-Stokes equations across multiple initial conditions concurrently by leveraging PyTorch tensor operations and `torchdiffeq`.

## Features

- **Batch Processing**: Solves the Navier-Stokes equations for multiple initial conditions simultaneously, vastly accelerating dataset generation and parametric studies.
- **Vorticity Formulation**: The solver uses the vorticity-stream function formulation of the 2D Navier-Stokes equations.
- **Differentiable Physics**: Written entirely in PyTorch using `torchdiffeq`, making the computation graph fully differentiable. It utilizes solvers such as `rk4` and `implicit_adams`.
- **Finite Difference Operators**: Custom-built, centralized matrices for first-order gradients (`dx`, `dy`) and Laplacians (`Lap`), optimized via fast matrix multiplications (`torch.mm`).
- **Poisson Solver integration**: Utilizes a customized Poisson equation solver to extract the stream function from the vorticity formulation field.
- **Gaussian Random Fields (GRFs)**: Built-in functionality to generate batched GRFs to be used as initial conditions. 

## Mathematical Formulation

The solver computes the 2D incompressible Navier-Stokes equations using the vorticity-stream function formulation. Given the vorticity field $\omega(x, y, t)$ and stream function $\psi(x, y, t)$, the equations are:

1. **Poisson Equation for Stream Function:**
   $$ \nabla^2 \psi = \omega $$

2. **Vorticity Transport Equation:**
   $$ \frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega - \left( \frac{\partial \psi}{\partial y} \frac{\partial \omega}{\partial x} - \frac{\partial \psi}{\partial x} \frac{\partial \omega}{\partial y} \right) + F $$

Where:
- $\nu$ is the kinematic viscosity.
- $( \frac{\partial \psi}{\partial y} \frac{\partial \omega}{\partial x} - \frac{\partial \psi}{\partial x} \frac{\partial \omega}{\partial y} )$ represents the advection/Jacobian term.
- $F$ is an applied external forcing term.


## Project Structure

- `NSE/prepare_dataset/solver.py`: Core logic for computing the right-hand side (RHS) of the PDE and advancing the solver over time using `torchdiffeq.odeint`.
- `NSE/prepare_dataset/operators.py`: Defines the `DifferentialOperators` class for calculating spatial derivatives (gradients and laplacians) using Finite Difference Methods.
- `NSE/prepare_dataset/PPT_solver.py`: Implements the Poisson equation solver (`solve_poisson`) to find the stream function.
- `NSE/prepare_dataset/prepare_data_nse.py`: Automation script to sample initial conditions via Gaussian Random Fields and build batched datasets of Navier-Stokes state evolutions.
- `lib/helper.py`: Provides utility functions like tensor upscaling and directory management.
- `lib/batchJacobian.pye`: Encrypted or specialized module to perform highly optimized batched Jacobian computations.

## Usage 

### Creating Datasets

The `prepare_data_nse.py` script serves as an entry point for testing the batch operations. 

```python
from NSE.prepare_dataset.prepare_data_nse import create_and_save_dataset, batch_grf

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_Sample = 10
Ns = 50

# Generate batched initial conditions using Gaussian Random Fields
initial_conditions = batch_grf(alpha=3.0, tau=0.5, s=Ns, A=-1.0, B=1.0, batch_size=n_Sample).to(device)

# Solve PDE and compile dataset
dataset = create_and_save_dataset(
    device, 
    initial_conditions, 
    nu=0.001, 
    dataset_segment_size=10, 
    N=Ns, 
    t0=0, 
    t_end=3, 
    steps=100
)
```

## Dependencies
- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `torchdiffeq`
- `pyfiglet`
- `sourcedefender` (Required to read `.pye` encrypted files)
