"""Loading and I/O: problems, models, influence scores, losses, LOO diffs."""

from pathlib import Path

import numpy as np
import torch

from pinnfluence.problem_factory import construct_problem
from pinnfluence.utils.models import ModelWrapper, PINNLoss


def load_problem(
    problem_name: str,
    params_dict: dict,
    soft_constrained: bool = True,
    float64: bool = True,
    seed: int = 0,
    drop_single_point_type: str = "none",
    load_path: str = None,
    force_reinitialize: bool = False,
):
    return construct_problem(
        problem_name=problem_name,
        layers=params_dict["layers"],
        num_domain=params_dict["num_domain"],
        num_boundary=params_dict["num_boundary"],
        num_initial=params_dict["num_initial"],
        n_iterations=params_dict["n_iterations"],
        n_iterations_lbfgs=params_dict["n_iterations_lbfgs"],
        optimizer=params_dict.get("optimizer", "adam"),
        soft_constrained=soft_constrained,
        float64=float64,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
        force_reinit=force_reinitialize,
    )


def get_X_y_true(model, num_points: int = 50_000):
    if model.data.soln is not None:
        X = model.data.geom.uniform_points(num_points)
        Y = model.data.soln(X)
    else:
        X = model.data.holdout_test_x
        Y = model.data.holdout_test_y
    return X, Y


def get_influence_scores(
    model_name: str,
    load_path: str,
    influence_load_path: str = None,
    right_term: str = "total_loss",
    left_term: str = "total_loss",
    method: str = "influences",
):
    """Load a single influence matrix from the per-term directory layout
    produced by ``precalculate_influences.main``.

    Files live in ``{model_name}_influence_scores/`` and are named
    ``{method}_{right_term}_{left_term}.npz`` (key ``scores``). ``method`` is
    ``"influences"`` for PINNfluence or ``"grad_dot"`` for the grad-dot baseline.
    """
    base = influence_load_path if influence_load_path is not None else load_path
    infl_dir = Path(base) / f"{model_name}_influence_scores"
    return np.load(infl_dir / f"{method}_{right_term}_{left_term}.npz")


def _influence_left_term(key: str, output_dim: int = 0) -> str:
    """Map an influence-component key to the ``left_term`` filename used by the
    per-term influence directory (right side fixed to ``total_loss``)."""
    return {
        "scores": "total_loss",
        "infl_scores_pde": "pde_loss",
        "infl_scores_bc": "bc_loss",
        "infl_scores_outputs": f"output_{output_dim}",
    }[key]


def get_leave_one_out_diff(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
):
    model, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    model_loo, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    test_x = model.data.holdout_test_x

    y_pred = model.predict(test_x)
    y_pred_loo = model_loo.predict(test_x)

    return y_pred - y_pred_loo


def get_removed_point_idx(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
):
    model_original, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    model_loo, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    return np.where(
        (model_original.data.train_x_all == model_loo.data.point_removed).all(axis=1)
    )[0][0]


def get_loss(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    target: str = "train",
):
    model, _, _, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    if target == "train":
        X = model.data.train_x_all
    else:
        X = model.data.holdout_test_x

    X_tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True)

    wrapped_model = ModelWrapper(model.net, model.data.pde, model.data.bcs)
    residuals = wrapped_model(X_tensor)
    loss_fn = PINNLoss()
    loss = loss_fn(residuals, torch.zeros(X.shape[0], 1))

    return loss

