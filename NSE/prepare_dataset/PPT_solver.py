import numpy as np
import torch

def solve_poisson(omega_batch, device, N, dx, dy):
    # Compute the forward FFT of the vorticity field for the whole batch
    omega_hat = torch.fft.fft2(omega_batch)

    # Generate wave numbers for a periodic domain
    kx = torch.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = torch.fft.fftfreq(N, d=dy) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    kx = kx.to(device)
    ky = ky.to(device)

    # Avoid division by zero at the zero frequency component
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1  # Set this to a non-zero value to avoid division by zero

    # Solve for stream function in the frequency domain
    psi_hat = -omega_hat / k_squared.unsqueeze(0)  # Expand k_squared to batch size

    # Inverse FFT to get the stream function in the physical domain
    psi = torch.fft.ifft2(psi_hat).real  # Take the real part
    return psi
