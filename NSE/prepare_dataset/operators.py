import torch
 
class DifferentialOperators:
    def __init__(self, N, dx, device):
        self.N = N
        self.dx = dx
        self.device = device
        self.Dx, self.Dy, self.Lap = self.create_matrices()

    def create_matrices(self):
        # Initialize matrices
        Dx = torch.zeros((self.N * self.N, self.N * self.N), device=self.device)
        Dy = torch.zeros((self.N * self.N, self.N * self.N), device=self.device)
        Lap = torch.zeros((self.N * self.N, self.N * self.N), device=self.device)

        for i in range(self.N):
            for j in range(self.N):
                index = i * self.N + j
                # Dx matrix
                if j == 0:
                    # Forward difference at the left boundary
                    Dx[index, index] = -1 / self.dx
                    Dx[index, index + 1] = 1 / self.dx
                elif j == self.N - 1:
                    # Backward difference at the right boundary
                    Dx[index, index] = 1 / self.dx
                    Dx[index, index - 1] = -1 / self.dx
                else:
                    # Central difference
                    Dx[index, index - 1] = -0.5 / self.dx
                    Dx[index, index + 1] = 0.5 / self.dx

                # Dy matrix
                if i == 0:
                    # Forward difference at the top boundary
                    Dy[index, index] = -1 / self.dx
                    Dy[index, index + self.N] = 1 / self.dx
                elif i == self.N - 1:
                    # Backward difference at the bottom boundary
                    Dy[index, index] = 1 / self.dx
                    Dy[index, index - self.N] = -1 / self.dx
                else:
                    # Central difference
                    Dy[index, index - self.N] = -0.5 / self.dx
                    Dy[index, index + self.N] = 0.5 / self.dx

                # Laplacian matrix
                Lap[index, index] = -4 / (self.dx ** 2)
                if j > 0:
                    Lap[index, index - 1] = 1 / (self.dx ** 2)
                if j < self.N - 1:
                    Lap[index, index + 1] = 1 / (self.dx ** 2)
                if i > 0:
                    Lap[index, index - self.N] = 1 / (self.dx ** 2)
                if i < self.N - 1:
                    Lap[index, index + self.N] = 1 / (self.dx ** 2)

        return Dx, Dy, Lap

    def grad(self, field):
        """Compute the gradient (dx, dy) using matrix multiplication"""
        field_flat = field.view(-1, self.N * self.N)
        dx = torch.mm(field_flat, self.Dx.t()).view(-1, self.N, self.N)
        dy = torch.mm(field_flat, self.Dy.t()).view(-1, self.N, self.N)
        return dx, dy

    def laplacian(self, field):
        """Compute the Laplacian using matrix multiplication"""
        field_flat = field.view(-1, self.N * self.N)
        lap_field = torch.mm(field_flat, self.Lap.t()).view(-1, self.N, self.N)
        return lap_field
