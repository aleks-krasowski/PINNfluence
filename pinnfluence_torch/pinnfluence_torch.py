"""Plain-PyTorch helpers for PINNfluence, to make use of the modified captum's
`ArnoldiIF`.

To use PINNfluence, one needs to write one residual module (an `nn.Module`),
i.e. a wrapper over all losses and the PINN itself.
It needs a forward method that receives the PINN input coordinates and
returns point-wise residuals as individual columns, more concretely:

* `forward(x: [B, d_in]) -> [B, n_terms]`, one column per loss term, where each
  row depends only on its own input row.
* Note that initial and boundary conditions (and other subset-specific terms)
  should be masked to zero elsewhere.
  In our implementation we apply the PDE residual to all points, but this
  may also optionally be masked to zero on non-relevant points.
* To keep every parameter involved in the computation, add
  `zero = u.sum() * 0` to every column.
* A `.net` attribute holding the network.
  This allows for access to the network and influence attribution
  directly on the outputs.
* PINNfluence can be sped up by using input derivatives via
  `torch.func.grad` to make use of batched input derivatives efficiently.
  Alternatively one can use `torch.autograd.grad` with
  `vectorize="batched_grad"` set in `ArnoldiInfluenceFunctionPrecomputed`.

Typical flow:

    # wrap the network with the residual wrapper
    residual_wrapper = MyResidualWrapper(net)

    IF = instantiate_precomputed_influence(
            residual_wrapper,
            x_train,                           # calculate Hessian using training data
            PINNLossWrapper()                  # another wrapper,
                                               # use per-sample, per-term loss
    )

    # calculate influence scores for the test data
    # in terms of the paper:
    # Inf(x_train, x_test)^{total_loss -> prediction_dim_0}
    scores = compute_influence_matrix(
            IF,
            residual_wrapper,
            "output_0",                 # specify the target loss/output term
                                        # to compute influence for
            "total_loss",               # specify the source loss/output term
                                        # in this case we check how training samples
                                        # influence the prediction of test points
                                        # w.r.t. the composite loss
            x_test                      # test data to compute influence for
    )

    # optionally: compute the fraction of the loss contributed by each term for the
    # chosen test data
    fracs = compute_loss_fractions(
            IF,
            residual_wrapper,
            x_test,
            train_terms={"pde": 0, "ic": 1, "bc": 2}    # index and name of each term
                                                        # in the loss vector returned by
                                                        # the residual_wrapper
    )

See `demo.ipynb` for a full example.
"""

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import torch
from captum.influence import ArnoldiInfluenceFunctionPrecomputed as ArnoldiIF
from torch import nn
from torch.utils.data import Dataset


# --- Loss and model wrappers -------------------------------------------------
class PINNLossWrapper(nn.Module):
    """Per-sample, per-term PINN loss on the [B, n_terms] residual-module output.

    Returns the [B, n_terms] element-wise losses captum's influence functions
    expect (it reads the `reduction` attribute and requires "none").

    Pass `include_all_losses=False, include_specific_ids=[...]` to use a subset
    of residual columns only.
    """

    def __init__(
        self,
        mode: Literal["mse", "l2", "l1"] = "mse",
        include_all_losses: bool = True,
        include_specific_ids: Iterable[int] | None = None,
        weights: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        if mode.lower() not in ("mse", "l2", "l1"):
            raise ValueError(f"mode must be mse|l2|l1, got {mode}")
        if include_all_losses == (include_specific_ids is not None):
            raise ValueError(
                "include_all_losses and include_specific_ids are mutually exclusive"
            )
        self.reduction = "none"  # captum reads this and requires "none"
        self.mode = mode.lower()
        self.include_all_losses = include_all_losses
        self.include_specific_ids = (
            list(include_specific_ids) if include_specific_ids is not None else None
        )
        self.weights = (
            torch.as_tensor(list(weights), dtype=torch.float64)
            if weights is not None
            else None
        )

    def forward(self, residuals: torch.Tensor, target=None) -> torch.Tensor:
        # `target` is the dummy label captum passes along; it is ignored
        if self.include_specific_ids is not None:
            residuals = residuals[:, self.include_specific_ids]
        if self.mode in ("mse", "l2"):
            losses = residuals**2
        else:
            losses = residuals.abs()
        if self.weights is not None:
            losses = losses * self.weights.to(losses)
        return losses


class NetPredWrapper(nn.Module):
    """Single output column of `net`, for influence on a prediction rather than
    a loss (used with `test_loss_fn=None`)."""

    def __init__(self, net: nn.Module, pred_idx: int = 0) -> None:
        super().__init__()
        self.net = net
        self.pred_idx = pred_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.net(x)
        if u.shape[1] > 1:
            u = u[:, self.pred_idx : self.pred_idx + 1]
        return u.view(-1, 1)


# --- Dataset -----------------------------------------------------------------
class DummyDataset(Dataset):
    """Wrapper that pairs inputs with zero targets:
    captum's influence APIs expect `(x, label)` tuples
    but the PINN loss ignores the label."""

    def __init__(self, x):
        tensor = torch.as_tensor(x, dtype=torch.get_default_dtype())
        self.x = tensor.detach().clone().requires_grad_(True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], torch.zeros_like(self.x[idx, 0:1])


# --- Term parsing ------------------------------------------------------------
def parse_influence_term(
    term: str | int | Sequence[int],
    residual_model: nn.Module,
) -> tuple[nn.Module, PINNLossWrapper | None]:
    """Map a term to the `(model, loss_fn)` pair for captum's IF.

    The residual columns are numbered by you, in the order your residual
    module's `forward` returns them. `output_N` reads `residual_model.net`.

    Vocabulary:
        "total_loss"        : all residual columns, MSE
        "output_N"          : N-th raw prediction column of `.net` (no loss)
        int                 : one residual column, MSE
        list | tuple of int : several residual columns, MSE
    """
    if term == "total_loss":
        return residual_model, PINNLossWrapper(mode="mse")

    if isinstance(term, int):
        term = [term]
    if isinstance(term, (list, tuple)):
        loss = PINNLossWrapper(include_all_losses=False, include_specific_ids=term)
        return residual_model, loss

    if isinstance(term, str) and term.startswith("output_"):
        try:
            idx = int(term.split("_", 1)[1])
        except ValueError as e:
            raise ValueError(f"invalid output term {term!r}; expected output_N") from e
        return NetPredWrapper(residual_model.net, pred_idx=idx), None

    raise ValueError(
        f"unknown loss term {term!r}; valid: 'total_loss', 'output_N', "
        f"a residual column index (int), or a list/tuple of column indices"
    )


# --- Influence functions -----------------------------------------------------
def instantiate_precomputed_influence(
    residual_model: nn.Module,
    x_train: np.ndarray,
    loss_fn: PINNLossWrapper | None = None,
    batch_size: int | None = None,
    seed: int = 0,
    show_progress: bool = False,
    r_cache_path: str | None = None,
    vectorize: Literal["vmap", "batched_grad"] | None = "vmap",
) -> ArnoldiIF:
    """Build a captum `ArnoldiIF` with the Hessian approximated on `loss_fn`
    (total per-sample PINN loss) over `x_train`.

    `r_cache_path` caches the Arnoldi basis to that file and reuses it on later
    runs (default `None`: no caching). `vectorize="vmap"` computes per-sample
    gradients with `torch.func.vmap`; `"batched_grad"` batches ordinary backward
    passes instead and is the one to use if your residual module calls
    `torch.autograd.grad`; `None` uses captum's per-sample loop (same result,
    much slower). Can also be changed post-init via `if_instance.vectorize`."""
    if loss_fn is None:
        loss_fn = PINNLossWrapper(mode="mse")
    train_set = DummyDataset(x_train)
    if batch_size is None:
        batch_size = min(1024, len(train_set))

    return ArnoldiIF(
        residual_model,
        train_dataset=train_set,
        loss_fn=loss_fn,
        checkpoint="dummy",  # net already loaded
        checkpoints_load_func=lambda *_: 0,  # net already loaded
        batch_size=batch_size,
        seed=seed,
        show_progress=show_progress,
        r_cache_path=r_cache_path,
        vectorize=vectorize,
    )


def compute_influence_matrix(
    if_instance: ArnoldiIF,
    residual_model: nn.Module,
    test_term: str | int | Sequence[int],
    train_term: str | int | Sequence[int],
    candidate_points: np.ndarray,
    batch_size: int = 1024,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute the `[n_candidate, n_train]` influence matrix for a given
    `(test_term, train_term)` pair on an already-built `if_instance`.

    `train_term` is the training-side loss term (the Hessian is always the one
    of the loss `if_instance` was built with); `test_term` is the test-side
    quantity, a loss term or a raw prediction (`output_N`).

    Sign follows the paper, `Inf = -grad_f^T H^-1 grad_L`: positive means
    upweighting the training point pushes the test term up.
    """
    if_instance.model, if_instance.loss_fn = parse_influence_term(
        train_term, residual_model
    )
    if_instance.model_test, if_instance.test_loss_fn = parse_influence_term(
        test_term, residual_model
    )
    if_instance.test_reduction_type = "none"

    candidate_set = DummyDataset(candidate_points)
    candidate_loader = torch.utils.data.DataLoader(
        candidate_set,
        batch_size=min(batch_size, len(candidate_set)),
    )

    # captum returns +grad_f^T H^-1 grad_L -> sign flip to match the paper's convention
    scores = -if_instance.influence(candidate_loader, show_progress=show_progress)
    return scores.detach().cpu().numpy().astype(np.float32)


# --- Loss-fraction decomposition ---------------------------------------------
def compute_loss_fractions(
    if_instance: ArnoldiIF,
    residual_model: nn.Module,
    candidate_points: np.ndarray,
    train_terms: dict[str, int | Sequence[int]],
    test_term: str = "output_0",
    batch_size: int = 1024,
    show_progress: bool = False,
) -> dict:
    """Decompose influence on ``test_term`` into fractional contributions from
    each training-side loss term, as in the single-seed branch of
    ``visualize_loss_fractions_lineplot`` in ``pinnfluence/utils/plotting.py``.

    ``train_terms`` maps a display name to residual column(s), e.g.
    ``{"pde": 0, "ic": 1, "bc": [2, 3]}``.

    For every test point we define:

    * ``fraction[term][i] = sum_j |score_term[i, j]| / denom[i]`` where
      ``denom[i] = sum_term sum_j |score_term[i, j]|``: share of the total
      absolute influence coming through each term.
    * ``coherence[i] = mean_j |sum_term score_term[i, j]| / sum_term |score_term[i, j]|``
     : per-test-point pointwise coherence of the signed contributions across
      terms. ``1 - coherence`` is the cancellation factor.

    Returns a dict with ``scores`` (raw per-term matrices) and ``fractions``,
    both keyed by the names in ``train_terms``, plus ``denom``, ``coherence``,
    ``candidate_points``, ``test_term``, and ``train_terms`` (tuple of names).
    """
    scores = {}
    for name, cols in train_terms.items():
        scores[name] = compute_influence_matrix(
            if_instance,
            residual_model,
            test_term=test_term,
            train_term=cols,
            candidate_points=candidate_points,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    eps = 1e-12
    signed_sum = np.zeros_like(next(iter(scores.values())))
    abs_sum = np.zeros_like(signed_sum)
    for s in scores.values():
        signed_sum += s
        abs_sum += np.abs(s)

    denom = np.maximum(
        np.abs(np.stack(list(scores.values()), axis=0)).sum(axis=(0, 2)), eps
    )
    fractions = {term: np.abs(s).sum(axis=1) / denom for term, s in scores.items()}
    coherence = (np.abs(signed_sum) / np.maximum(abs_sum, eps)).mean(axis=1)

    return {
        "scores": scores,
        "fractions": fractions,
        "denom": denom,
        "coherence": coherence,
        "candidate_points": candidate_points,
        "test_term": test_term,
        "train_terms": tuple(train_terms),
    }
