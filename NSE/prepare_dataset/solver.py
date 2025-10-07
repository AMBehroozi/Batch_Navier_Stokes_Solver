from PPT_solver import solve_poisson
from torchdiffeq import odeint

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
        method='rk4'
    )
    return w_flat_sol