"""Demo helpers that are not part of PINNfluence itself: the tanh MLP trained
in `demo.ipynb` and plots for the influence results of `pinnfluence_torch.py`."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import init


# --- Network -----------------------------------------------------------------
class Net(nn.Module):
    """Tanh MLP. Layers are stored as `layer1 ... layerN` (the key names in the
    saved state dicts)."""

    def __init__(
        self,
        layer_sizes: Sequence[int] | None = None,
        use_input_transform: bool = True,
        x_end: float = 1.0,
        t_end: float = 1.0,
    ) -> None:
        super().__init__()

        self.layer_sizes = (
            list(layer_sizes) if layer_sizes is not None else [2, 64, 64, 64, 1]
        )
        self.use_input_transform = use_input_transform
        self.x_end = x_end
        self.t_end = t_end

        for i in range(len(self.layer_sizes) - 1):
            setattr(
                self,
                f"layer{i + 1}",
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]),
            )

        # match DeepXDE init
        for i in range(1, len(self.layer_sizes)):
            init.xavier_normal_(getattr(self, f"layer{i}").weight)
        init.zeros_(getattr(self, f"layer{len(self.layer_sizes) - 1}").bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_transform:
            x = feature_transform(x, self.x_end, self.t_end)

        for i in range(1, len(self.layer_sizes) - 1):
            x = torch.tanh(getattr(self, f"layer{i}")(x))
        return getattr(self, f"layer{len(self.layer_sizes) - 1}")(x)


def feature_transform(
    x: torch.Tensor, x_end: float = 1.0, t_end: float = 1.0
) -> torch.Tensor:
    """Scale (x, t) to [0, 1]^2 (domain starts at x = t = 0)."""
    return torch.cat([x[:, 0:1] / x_end, x[:, 1:2] / t_end], dim=1)


# --- Plotting ----------------------------------------------------------------
def plot_loss_fractions_lineplot(
    result: dict,
    dim: int = 1,
    num_bins: int = 50,
    ax: plt.Axes | None = None,
    dim_label: str | None = None,
    show_cancellation: bool = True,
) -> plt.Axes:
    """Line-plot loss-term fractions (and cancellation factor) binned along
    one coordinate of the candidate points. Consumes the dict returned by
    ``compute_loss_fractions``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    coord = result["candidate_points"][:, dim]
    order = np.argsort(coord)
    coord_s = coord[order]

    bins = np.linspace(coord_s.min(), coord_s.max(), num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def _bin(values: np.ndarray) -> np.ndarray:
        v = values[order]
        out = np.full(num_bins, np.nan)
        for i in range(num_bins):
            upper_op = np.less_equal if i == num_bins - 1 else np.less
            mask = (coord_s >= bins[i]) & upper_op(coord_s, bins[i + 1])
            if mask.any():
                out[i] = v[mask].mean()
        return out

    if show_cancellation:
        cancellation = 1.0 - result["coherence"]
        ax.plot(
            centers,
            _bin(cancellation),
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"cancellation $\\kappa$ (mean: {cancellation.mean():.3f})",
        )

    for term, frac in result["fractions"].items():
        ax.plot(
            centers,
            _bin(frac),
            linewidth=2,
            label=f"{term} (mean: {frac.mean():.3f})",
        )

    ax.set_xlabel(dim_label or f"dimension {dim}")
    ax.set_ylabel("loss fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{result['test_term']} <- " + " , ".join(map(str, result["train_terms"])))
    return ax


def plot_point_influence(
    train_x: np.ndarray,
    candidate_x: np.ndarray,
    scores: np.ndarray,
    test_idx: int,
    ax: plt.Axes | None = None,
    cmap: str = "bwr",
    symmetric: bool = True,
    title: str | None = None,
):
    """Scatter training points coloured by their influence on one test point.

    The test point itself is marked with a black x. `scores` is the
    `[n_candidate, n_train]` matrix returned by `compute_influence_matrix`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    row = scores[test_idx]
    if symmetric:
        lim = float(np.abs(row).max()) or 1.0
        kwargs = dict(vmin=-lim, vmax=lim)
    else:
        kwargs = {}

    sc = ax.scatter(train_x[:, 0], train_x[:, 1], c=row, cmap=cmap, s=8, **kwargs)
    ax.scatter(
        candidate_x[test_idx, 0],
        candidate_x[test_idx, 1],
        marker="x",
        c="black",
        s=160,
        linewidths=2,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title is not None:
        ax.set_title(title)
    plt.colorbar(sc, ax=ax)
    return ax
