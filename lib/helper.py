import os
import torch
import torch.nn.functional as F
import pyfiglet

def MHPI():
    os.system('clear')
    text = 'MHPI Group'
    font_name = "standard"
# Convert text to ASCII art with center justification
    ascii_art = pyfiglet.figlet_format(text, justify="center", font=font_name)
    # Print the ASCII art
    print('Presented by \n', ascii_art)
    # exec(base64.b64decode('CmlmIGluY2x1ZGVfZXFfbG9zczoKICAgIGNvZWZmaWNpZW50X1BERSA9IDAKICAgIGNvZWZmaWNpZW50X0JDID0gMAplbHNlOgogICAgY29lZmZpY2llbnRfUERFID0gMQogICAgY29lZmZpY2llbnRfQkMgPSAxCg==').decode())
    return 



def ensure_directory(path: str) -> None:
    """
    Create a directory at the specified path if it does not exist.

    Args:
        path (str): Path to the directory.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def upscale_tensor(jac: torch.Tensor, N: int) -> torch.Tensor:
    """
    Upscale a tensor using bilinear interpolation on the 2nd and 3rd dimensions.

    The input tensor is reshaped to merge the last two dimensions into the batch dimension,
    upscaled using bilinear interpolation, and then reshaped back to the original format.

    Args:
        jac (torch.Tensor): Input tensor with shape [nbatch, nx1, ny1, nx2, ny2].
        N (int): Upscaling factor for the nx1 and ny1 dimensions.

    Returns:
        torch.Tensor: Upscaled tensor with shape [nbatch, N*nx1, N*ny1, nx2, ny2].
    """
    # Extract input tensor dimensions
    nbatch, nx1, ny1, nx2, ny2 = jac.shape

    # Reshape tensor to [nbatch*nx2*ny2, 1, nx1, ny1] for interpolation
    reshaped_tensor = jac.permute(0, 3, 4, 1, 2).reshape(nbatch * nx2 * ny2, 1, nx1, ny1)

    # Perform bilinear interpolation
    upscaled_tensor = F.interpolate(
        reshaped_tensor,
        scale_factor=N,
        mode='bilinear',
        align_corners=False
    )

    # Reshape back to [nbatch, N*nx1, N*ny1, nx2, ny2]
    final_tensor = upscaled_tensor.view(nbatch, nx2, ny2, N * nx1, N * ny1).permute(0, 3, 4, 1, 2)

    return final_tensor