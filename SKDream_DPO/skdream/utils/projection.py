"""Small camera-matrix helpers used by data preparation and inference."""

import numpy as np
import torch


def perspective(
    fovy: float = 0.7854,
    aspect: float = 1.0,
    near: float = 0.1,
    far: float = 1000.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    y = np.tan(fovy / 2)
    return torch.tensor(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )


def translate(
    x: float, y: float, z: float, device: torch.device | str | None = None
) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_x(angle: float, device: torch.device | str | None = None) -> torch.Tensor:
    sine, cosine = np.sin(angle), np.cos(angle)
    return torch.tensor(
        [[1, 0, 0, 0], [0, cosine, sine, 0], [0, -sine, cosine, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_y(angle: float, device: torch.device | str | None = None) -> torch.Tensor:
    sine, cosine = np.sin(angle), np.cos(angle)
    return torch.tensor(
        [[cosine, 0, sine, 0], [0, 1, 0, 0], [-sine, 0, cosine, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
