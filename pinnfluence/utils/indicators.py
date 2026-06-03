"""Influence-based indicators: correlations, directionality, regional, proximity."""


import numpy as np
import scipy.stats as stats

from pinnfluence.utils.common import (
    get_min_max_from_geom,
    scale_x,
    scaled_rbf_kernel,
)
from pinnfluence.utils.io import (
    _influence_left_term,
    get_influence_scores,
    get_leave_one_out_diff,
    get_removed_point_idx,
    load_problem,
)


def get_loo_correlation(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    drop_single_point_type: str = "IC",
    load_path: str = None,
    influence_load_path: str = None,
):
    _, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    loo_error = get_leave_one_out_diff(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    removed_point_idx = get_removed_point_idx(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        drop_single_point_type=drop_single_point_type,
        load_path=load_path,
    )

    spearman_corr = []
    pearson_corr = []

    for i in range(loo_error.shape[1]):
        # influence of training points w.r.t. output i (right side = total loss)
        infl_output_i = get_influence_scores(
            model_name=model_name,
            load_path=load_path,
            influence_load_path=influence_load_path,
            left_term=f"output_{i}",
        )["scores"]
        spearman_corr.append(
            stats.spearmanr(
                infl_output_i[:, removed_point_idx],
                loo_error[:, i].flatten(),
            )
        )
        pearson_corr.append(
            stats.pearsonr(
                infl_output_i[:, removed_point_idx],
                loo_error[:, i].flatten(),
            )
        )

    return spearman_corr, pearson_corr


def get_directionality_indicator(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    direction_dimension: int = 0,
    use_radial_distances: bool = False,
    right_term: str = "total_loss",
    left_term: str = "total_loss",
    method: str = "influences",
):
    """
    Calculate directionality indicator scores.

    For each test point, compute the ratio of influence scores from training points
    that are "upstream" in the specified direction dimension.

    Args:
        problem_name: Name of the problem
        params_dict: Problem parameters
        seed: Random seed
        load_path: Path to load models from
        direction_dimension: Which dimension to use for directionality (0 or 1)
        use_radial_distances: Whether to use radial distances instead of linear dimension
        right_term: Right side term for influence file (e.g., 'total_loss', 'pde_loss', 'output_0')
        left_term: Left side term for influence file (e.g., 'total_loss', 'pde_loss', 'bc_0')

    Returns:
        dir_indicator_scores: List of directionality indicator scores for each test point
        point_ratios: List of point ratios for each test point
    """
    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    # Load influence/grad_dot file with new naming convention
    _infl_base = influence_load_path if influence_load_path is not None else load_path
    infl_path = _infl_base.joinpath(f"{model_name}_influence_scores")
    infl_file = infl_path / f"{method}_{right_term}_{left_term}.npz"

    if not infl_file.exists():
        raise FileNotFoundError(f"Influence file not found: {infl_file}")
    else:
        print(f"Loading influence scores from: {infl_file}")

    infl_data = np.load(infl_file)
    test_x = infl_data["candidate_points"]
    influences_cur = infl_data["scores"]

    train_x = model.data.train_x_all

    print(f"train_x shape: {train_x.shape}")
    print(f"candidates shape: {test_x.shape}")
    print(f"influences shape: {influences_cur.shape}")

    boundary_mask = model.data.geom.on_boundary(test_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask = np.logical_or(boundary_mask, model.data.geom.on_initial(test_x))

    dir_indicator_scores = []
    point_ratios = []

    for i, x in enumerate(test_x):
        if boundary_mask[i]:
            dir_indicator_scores.append(np.nan)
            point_ratios.append(np.nan)
            continue

        if use_radial_distances:
            center = np.array([0, 0])
            indices = np.where(
                np.linalg.norm(train_x - center, axis=1) <= np.linalg.norm(x - center)
            )[0]
        else:
            indices = np.where(
                train_x[:, direction_dimension] <= x[direction_dimension]
            )[0]

        total_abs_influence = np.abs(influences_cur[i, :]).sum()
        if total_abs_influence > 0:
            upstream_influence = np.abs(influences_cur[i, indices]).sum()
            dir_indicator_scores.append(upstream_influence / total_abs_influence)
        else:
            dir_indicator_scores.append(np.nan)

        point_ratios.append(len(indices) / len(train_x))

    return dir_indicator_scores, point_ratios


def get_correlation(x, y):
    return stats.pearsonr(x, y), stats.spearmanr(x, y)


def get_influences_for_closest_point(
    point: np.ndarray,
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    target: str = "test",
    target_key: str = "infl_scores_outputs",
    target_dimension: int = 0,
):
    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    if target == "train":
        X = model.data.train_x_all
    else:
        X = model.data.holdout_test_x

    target_influences = get_influence_scores(
        model_name=model_name,
        load_path=load_path,
        influence_load_path=influence_load_path,
        left_term=_influence_left_term(target_key, target_dimension),
    )["scores"]
    target_influences = target_influences * len(model.data.train_x_all)

    distances = np.linalg.norm(X - point, axis=1)
    closest_point_idx = np.argmin(distances)

    if target == "train":
        return target_influences[:, closest_point_idx], X[closest_point_idx]
    else:
        return target_influences[closest_point_idx], X[closest_point_idx]


def get_proximity_indicator(
    problem_name: str,
    params_dict: dict,
    seed: int = 0,
    load_path: str = None,
    influence_load_path: str = None,
    sigma: float = 1.0,
    influence_key: str = "scores",
):
    """
    Calculate proximity indicator scores using scaled RBF kernel.

    For each test point, compute the weighted sum of influence scores from training points,
    where weights are given by the scaled RBF kernel based on distance.

    Args:
        problem_name: Name of the problem
        params_dict: Problem parameters
        seed: Random seed
        load_path: Path to load models from
        sigma: RBF kernel bandwidth parameter
        influence_key: Which influence scores to use ('scores', 'infl_scores_pde', 'infl_scores_bc', 'infl_scores_outputs')

    Returns:
        proximity_scores: List of proximity indicator scores for each test point
    """
    from functools import partial

    model, _, model_name, _ = load_problem(
        problem_name=problem_name,
        params_dict=params_dict,
        seed=seed,
        load_path=load_path,
    )

    train_x = model.data.train_x_all
    test_x = model.data.holdout_test_x

    # Get geometry bounds for scaling
    x_min, x_max = get_min_max_from_geom(model.data.geom)

    boundary_mask_test = model.data.geom.on_boundary(test_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask_test = np.logical_or(
            boundary_mask_test, model.data.geom.on_initial(test_x)
        )

    boundary_mask_train = model.data.geom.on_boundary(train_x)
    if hasattr(model.data.geom, "on_initial"):
        boundary_mask_train = np.logical_or(
            boundary_mask_train, model.data.geom.on_initial(train_x)
        )

    # Create partial function for scaled RBF kernel
    scale_x_partial = partial(scale_x, x_min=x_min, x_max=x_max)
    scaled_rbf_kernel_ = partial(
        scaled_rbf_kernel, scale_fn=scale_x_partial, sigma=sigma
    )

    proximity_scores = []

    # Handle different influence keys
    if influence_key == "infl_scores_outputs":
        # For outputs, we need to handle multiple dimensions
        num_outputs = model.net.linears[-1].out_features
        for output_dim in range(num_outputs):
            # Calculate for each output dimension
            output_scores = []
            influences_cur = get_influence_scores(
                model_name=model_name,
                load_path=load_path,
                influence_load_path=influence_load_path,
                left_term=f"output_{output_dim}",
            )["scores"][:, ~boundary_mask_train]

            for i, x in enumerate(test_x):
                if boundary_mask_test[i]:
                    proximity_scores.append(np.nan)
                    continue

                proximity_weights = scaled_rbf_kernel_(train_x[~boundary_mask_train], x)

                weighted_influence = np.abs(
                    influences_cur[i, :] * proximity_weights
                ).sum()
                total_abs_influence = np.abs(influences_cur[i, :]).sum()

                if total_abs_influence > 0:
                    output_scores.append(weighted_influence / total_abs_influence)
                else:
                    output_scores.append(np.nan)

            # Average across output dimensions
            proximity_scores.append(np.mean(output_scores))
    else:
        # For other keys, handle as single dimension
        influences_cur = get_influence_scores(
            model_name=model_name,
            load_path=load_path,
            influence_load_path=influence_load_path,
            left_term=_influence_left_term(influence_key),
        )["scores"]

        influences_cur = influences_cur[:, ~boundary_mask_train]

        for i, x in enumerate(test_x):
            if boundary_mask_test[i]:
                proximity_scores.append(np.nan)
                continue

            proximity_weights = scaled_rbf_kernel_(train_x[~boundary_mask_train], x)
            weighted_influence = np.abs(influences_cur[i, :] * proximity_weights).sum()
            total_abs_influence = np.abs(influences_cur[i, :]).sum()

            if total_abs_influence > 0:
                proximity_scores.append(weighted_influence / total_abs_influence)
            else:
                proximity_scores.append(np.nan)

    return proximity_scores


def region_fraction(cur_infl, test_mask, train_mask, eps=1e-12):
    """
    Calculate the fraction of influence from train_mask region to test_mask region.

    Args:
        cur_infl: Influence matrix (test_points x train_points)
        test_mask: Boolean mask for test points
        train_mask: Boolean mask for train points
        eps: Small epsilon to avoid division by zero

    Returns:
        float: Fraction of influence from train region to test region
    """
    # num = np.abs(cur_infl[test_mask][:, train_mask]).sum()
    # den = np.abs(cur_infl[test_mask]).sum() + eps
    # return float(num/den)
    # macro average
    rows = cur_infl[np.asarray(test_mask)]  # select B
    if rows.shape[0] == 0:
        return 0.0  # or float('nan') if you prefer signaling "no test points"

    num = np.abs(rows[:, np.asarray(train_mask)]).sum(axis=1)  # per-row numerator
    den = np.abs(rows).sum(axis=1) + eps  # per-row denominator
    return float(np.mean(num / den))

