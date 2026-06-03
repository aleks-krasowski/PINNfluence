"""Small shared helpers: stdout capture, device setup, geometry/kernel utils."""

import os
import sys
from io import StringIO

import numpy as np
import torch


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def set_default_device(device: str = "cpu"):
    if device == "cpu":
        torch.set_default_device("cpu")
        print("Using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available. Using CPU")
            torch.set_default_device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch._dynamo.disable()
            torch.set_default_device("mps")
            torch._dynamo.reset()
            print("Using mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS")
        else:
            print("MPS not available. Using CPU")
            torch.set_default_device("cpu")
    else:
        print("Invalid device. Using CPU")
        torch.set_default_device("cpu")


def get_min_max_from_geom(geom):
    if hasattr(geom, "timedomain"):
        x1_min, x1_max = geom.geometry.bbox
        x2_min, x2_max = geom.timedomain.bbox
        x_min = np.concatenate([x1_min, x2_min])
        x_max = np.concatenate([x1_max, x2_max])
    else:
        x_min, x_max = geom.bbox
    return x_min, x_max


def scale_x(x, x_min=0, x_max=1):
    return (x - x_min) / (x_max - x_min)


def scaled_rbf_kernel(X, Y, sigma=1, scale_fn=scale_x):
    return np.exp(
        -np.linalg.norm(scale_fn(X) - scale_fn(Y), ord=2, axis=1) / (2 * sigma**2)
    )

