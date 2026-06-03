"""Figure generation: prediction/influence heatmaps and loss-fraction plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

from pinnfluence.utils.common import Capturing
from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
from pinnfluence.utils.io import get_X_y_true, load_problem
from pinnfluence.utils.models import ModelWrapper, PINNLoss


loss_term_names = {
    "allen_cahn": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=1$)",
        "bc_2": "Dirichlet BC ($x=-1$)",
        "output_0": "$\\hat u$",
    },
    "burgers": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=-1$ and $x=1$)",
        "output_0": "$\\hat u$",
    },
    "diffusion": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Dirichlet BC ($x=-1$ and $x=1$)",
        "output_0": "$\\hat u$",
    },
    "drift_diffusion": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "IC Loss",
        "bc_1": "Periodic BC ($x=0$)",
        "bc_2": "Periodic BC ($x=2\\pi$)",
        "output_0": "$\\hat u$",
    },
    "navier_stokes_nd": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE (continuity)",
        "pde_1": "PDE (x-momentum)",
        "pde_2": "PDE (y-momentum)",
        "bc_loss": "BC Loss",
        "bc_0": "No-slip $u$ BC (x-direction)",
        "bc_1": "No-slip $v$ BC (y-direction)",
        "bc_2": "Inflow $u$ BC (x-direction)",
        "bc_3": "Inflow $v$ BC (y-direction)",
        "bc_4": "Outflow $u$ BC (x-direction)",
        "bc_5": "Outflow $v$ BC (y-direction)",
        "output_0": "$\\hat u$",
        "output_1": "$\\hat v$",
        "output_2": "$\\hat p$",
    },
    "poisson_disk": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": "Dirichlet BC ($u(x) = 0$ at $||{x}||=1$)",
        "output_0": "$\\hat u$",
    },
    "wave": {
        "total_loss": "Total Loss",
        "pde_loss": "PDE Loss",
        "pde_0": "PDE Loss",
        "bc_loss": "BC Loss",
        "bc_0": f"IC Loss",
        "bc_1": "Dirichlet BC ($u(0,t) = 0$)",
        "bc_2": "Dirichlet BC ($u(1,t) = 0$)",
        "bc_3": f"Operator BC ($\\frac{{\\partial u}}{{\\partial t}}$ at $t=0$)",
        "output_0": "$\\hat u$",
    },
}


def plot_prediction_heatmap(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str,
    cmap: str = "jet",
):
    if y_true.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3)

        sc = ax[0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap)
        ax[0].set_title("Predicted")
        fig.colorbar(sc, ax=ax[0])
        sc = ax[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap)
        ax[1].set_title("True")
        fig.colorbar(sc, ax=ax[1])
        sc = ax[2].scatter(X[:, 0], X[:, 1], c=residuals, cmap=cmap)
        ax[2].set_title("Residuals")
        fig.colorbar(sc, ax=ax[2])

    else:
        fig, ax = plt.subplots(
            figsize=(15, 5 * y_true.shape[1]),
            ncols=3,
            nrows=y_true.shape[1],
            sharex=True,
            sharey=True,
        )

        ax[0, 0].set_title("Predicted")
        ax[0, 1].set_title("True")
        ax[0, 2].set_title("Residuals")

        for i in range(y_true.shape[1]):
            sc = ax[i, 0].scatter(X[:, 0], X[:, 1], c=y_pred[:, i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 0])
            sc = ax[i, 1].scatter(X[:, 0], X[:, 1], c=y_true[:, i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 1])
            sc = ax[i, 2].scatter(X[:, 0], X[:, 1], c=residuals[i], cmap=cmap)
            fig.colorbar(sc, ax=ax[i, 2])

    fig.suptitle(title)

    return fig


def plot_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    title: str = None,
    cmap: str = "bwr",
    use_norm: bool = False,
    x_label: str = "x1",
    y_label: str = "x2",
    figsize: tuple = (10, 10),
):
    if use_norm:
        norm = TwoSlopeNorm(vmin=-np.max(np.abs(y)), vcenter=0, vmax=np.max(np.abs(y)))
    else:
        norm = None

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm)
    fig.colorbar(sc, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def visualize_predictions_comparison(
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    figsize=(14, 6),
    model_zoo_base="../../model_zoo_cluster",
    show_loss=False,
    show_error=False,
    use_train_points=False,
    cmap="coolwarm",
    marker_sizes=[50, 50],
    logscale=False,
):
    """
    Visualize model predictions across different model configurations.

    Parameters:
    -----------
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo
    show_residuals : bool
        If True, show residuals (y_true - y_pred) instead of predictions
    show_error : bool
        If True, show absolute error |y_true - y_pred| instead of predictions
        Takes precedence over show_residuals if both are True

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import load_problem

    configs = {"good": ("PROBLEMS", PROBLEMS), "bad": ("BAD_PROBLEMS", BAD_PROBLEMS)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(
        configs.items()
    ):
        ax = axes[idx]

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"

        # Determine which seeds to use
        if seed == "all":
            if "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points
            if seed == "all":
                # Load first model to get geometry
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                    )

                # Get initial X
                if use_train_points:
                    X_orig = model.data.train_x_all
                    if model.data.soln is not None:
                        y_true = model.data.soln(X_orig)
                    else:
                        if show_error:
                            raise ValueError(
                                "Ground truth solution not available for training points."
                            )
                else:
                    X_orig, y_true = get_X_y_true(model)

                # Identify boundary and initial points
                boundary_mask = data.geom.on_boundary(X_orig)
                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(X_orig)
                else:
                    initial_mask = np.zeros(len(X_orig), dtype=bool)

                num_boundary = boundary_mask.sum()
                num_initial = initial_mask.sum()

                # Create uniform boundary and initial points
                # These are generated directly from geometry, ensuring consistent spatial locations
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, X_orig.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, X_orig.shape[1]))

                # Domain points (non-boundary, non-initial) from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = X_orig[domain_mask]

                # Create reference X with uniform boundary/initial points
                # For each seed, predictions/losses will be computed at these exact locations
                X = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Recompute y_true at reference points if needed
                if model.data.soln is not None:
                    y_true = model.data.soln(X)
                elif show_error:
                    # Interpolate y_true to reference points
                    from scipy.interpolate import NearestNDInterpolator

                    interp = NearestNDInterpolator(X_orig, y_true)
                    y_true = interp(X)

                # Collect predictions from all seeds
                all_predictions = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if show_loss:
                        if use_train_points:
                            # For loss at training points, find closest training points and compute loss there
                            train_x_s = model.data.train_x_all

                            # Compute loss at all training points
                            X_train_tensor = torch.tensor(
                                train_x_s, dtype=torch.float64, requires_grad=True
                            )
                            wrapped_model = ModelWrapper(
                                model.net, model.data.pde, model.data.bcs
                            )
                            residuals = wrapped_model(X_train_tensor)
                            loss_fn = PINNLoss()
                            loss_all = loss_fn(
                                residuals, torch.zeros(train_x_s.shape[0], 1)
                            ).sum(axis=1)
                            loss_all = loss_all.detach().numpy()

                            # Map losses to reference points
                            y_pred = np.zeros(len(X))

                            # Domain points: direct copy
                            boundary_mask_s = data.geom.on_boundary(train_x_s)
                            if hasattr(data.geom, "on_initial"):
                                initial_mask_s = data.geom.on_initial(train_x_s)
                            else:
                                initial_mask_s = np.zeros(len(train_x_s), dtype=bool)
                            domain_mask_s = ~(boundary_mask_s | initial_mask_s)

                            y_pred[: len(domain_points)] = loss_all[domain_mask_s]

                            # Boundary points: find closest training boundary point
                            boundary_points_s = train_x_s[boundary_mask_s]
                            boundary_loss_s = loss_all[boundary_mask_s]

                            for i, ref_point in enumerate(uniform_boundary):
                                distances = np.linalg.norm(
                                    boundary_points_s - ref_point, axis=1
                                )
                                closest = np.argmin(distances)
                                y_pred[len(domain_points) + i] = boundary_loss_s[
                                    closest
                                ]

                            # Initial points: find closest training initial point
                            if num_initial > 0:
                                initial_points_s = train_x_s[initial_mask_s]
                                initial_loss_s = loss_all[initial_mask_s]

                                for i, ref_point in enumerate(uniform_initial):
                                    distances = np.linalg.norm(
                                        initial_points_s - ref_point, axis=1
                                    )
                                    closest = np.argmin(distances)
                                    y_pred[
                                        len(domain_points) + len(uniform_boundary) + i
                                    ] = initial_loss_s[closest]

                            y_pred = y_pred.reshape(-1, 1)
                        else:
                            # For loss at test points, compute directly at reference points
                            X_tensor = torch.tensor(
                                X, dtype=torch.float64, requires_grad=True
                            )
                            wrapped_model = ModelWrapper(
                                model.net, model.data.pde, model.data.bcs
                            )
                            residuals = wrapped_model(X_tensor)
                            loss_fn = PINNLoss()
                            y_pred = loss_fn(residuals, torch.zeros(X.shape[0], 1)).sum(
                                axis=1
                            )
                            y_pred = y_pred.detach().numpy().reshape(-1, 1)
                    else:
                        # For predictions, compute directly at reference points
                        y_pred = model.predict(X)

                    # for navier stokes simply get the first output component
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 0].reshape(-1, 1)

                    if show_error:
                        y_pred = np.abs(y_true - y_pred)

                    all_predictions.append(y_pred)

            else:
                # Single seed case
                all_predictions = []
                X = None
                y_true = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if use_train_points:
                        X = model.data.train_x_all
                        if model.data.soln is not None:
                            y_true = model.data.soln(X)
                        else:
                            if show_error:
                                raise ValueError(
                                    "Ground truth solution not available for training points."
                                )

                    if X is None:
                        X, y_true = get_X_y_true(model)

                    if show_loss:
                        X_tensor = torch.tensor(
                            X, dtype=torch.float64, requires_grad=True
                        )
                        wrapped_model = ModelWrapper(
                            model.net, model.data.pde, model.data.bcs
                        )
                        residuals = wrapped_model(X_tensor)
                        loss_fn = PINNLoss()
                        y_pred = loss_fn(residuals, torch.zeros(X.shape[0], 1)).sum(
                            axis=1
                        )
                        y_pred = y_pred.detach().numpy().reshape(-1, 1)
                    else:
                        y_pred = model.predict(X)

                    # for navier stokes simply get the first output component
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 0].reshape(-1, 1)

                    if show_error:
                        y_pred = np.abs(y_true - y_pred)

                    all_predictions.append(y_pred)

            # Average predictions across all seeds
            y_pred = np.mean(all_predictions, axis=0)

            # Determine what to plot
            if show_error:
                plot_data = y_pred
                label = "Absolute Error"
                title_suffix = " (error)"
            elif show_loss:
                plot_data = y_pred
                label = "Residual"
                title_suffix = " (residuals)"
                norm = None
            else:
                plot_data = y_pred
                label = "Predicted Value"
                title_suffix = ""
                norm = None
            if logscale:
                norm = LogNorm()

            sc = ax.scatter(
                *X.T,
                c=plot_data,
                cmap=cmap,
                s=marker_sizes[idx],
                alpha=0.7,
                norm=norm if show_loss else None,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(
                    f"{config_name} (avg {len(all_predictions)} seeds){title_suffix}"
                )
            else:
                ax.set_title(f"{config_name} (seed={seed}){title_suffix}")

            ax.set_xlabel("x")
            if idx == 0:
                ax.set_ylabel("t")

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(label)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            if seed == "all":
                ax.set_title(f"{config_name}")
            else:
                ax.set_title(f"{config_name} (seed={seed})")
    return fig, axes


def visualize_influence_comparison(
    target_coords,
    left_term,
    right_term,
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    use_test_points=True,
    figsize=(8, 6),
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    marker_sizes=[50, 50],
    marker_size_cross=300,
    absolute=True,
    alpha=None,
):
    """
    Visualize influences for a target point across different model configurations.

    Parameters:
    -----------
    target_coords : tuple
        (x, y) coordinates of the target point
    left_term : str
        Left side term for influence file (e.g., 'total_loss', 'pde_loss', 'bc_0')
    right_term : str
        Right side term for influence file (e.g., 'total_loss', 'output_0')
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    use_test_points : bool
        If True, find closest point in test set; if False, in training set
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo

    Returns:
    --------
    fig_good, fig_bad : tuple of matplotlib figures
        Two separate figures, one for the "good" configuration and one for the "bad" configuration
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import CenteredNorm

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import load_problem

    configs = {"good": ("PROBLEMS", PROBLEMS), "bad": ("BAD_PROBLEMS", BAD_PROBLEMS)}

    # Create separate figures for each configuration
    figures = {}
    axes_dict = {}

    target_point = np.array(target_coords)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(
        configs.items()
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        figures[config_name] = fig
        axes_dict[config_name] = ax

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"
        infl_load_path = (
            Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"
        )

        # Determine which seeds to use
        if seed == "all":
            if "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points for train set
            if seed == "all":
                # Load first model to get geometry
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                    )

                # Count boundary and initial points in training set
                train_x_first = data.train_x_all
                boundary_mask = data.geom.on_boundary(train_x_first)
                num_boundary = boundary_mask.sum()

                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(train_x_first)
                    num_initial = initial_mask.sum()
                else:
                    num_initial = 0
                    initial_mask = np.zeros(len(train_x_first), dtype=bool)

                # Create uniform reference points
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, train_x_first.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, train_x_first.shape[1]))

                # Domain points from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = train_x_first[domain_mask]

                # Reference training set
                train_x = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Load influence file from first seed to get test points
                infl_path = infl_load_path.joinpath(f"{model_name}_influence_scores")
                infl_file = infl_path.joinpath(
                    f"influences_{right_term}_{left_term}.npz"
                )

                if not infl_file.exists():
                    ax.text(
                        0.5,
                        0.5,
                        f"Missing:\ninfluences_{right_term}_{left_term}.npz",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="red",
                    )
                    ax.set_title(f"{config_name.capitalize()}")
                    continue

                infl_data = np.load(infl_file)
                test_x = infl_data["candidate_points"]

                # Find closest point to target
                if use_test_points:
                    points_to_search = test_x
                    point_type = "test"
                else:
                    points_to_search = train_x
                    point_type = "train"

                distances = np.linalg.norm(points_to_search - target_point, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = points_to_search[closest_idx]

                # Collect mapped influences from all seeds
                all_influences = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    infl_path = infl_load_path.joinpath(
                        f"{model_name}_influence_scores"
                    )
                    infl_file = infl_path.joinpath(
                        f"influences_{right_term}_{left_term}.npz"
                    )

                    if not infl_file.exists():
                        continue

                    infl_data = np.load(infl_file)
                    scores = infl_data["scores"] * 1 / len(data.train_x_all)
                    train_x_s = data.train_x_all

                    # Get influences for the closest point
                    if use_test_points:
                        influences_s = scores[closest_idx, :]

                        # Map influences to reference training points (only needed for use_test_points=True)
                        mapped_influences = np.zeros(len(train_x))

                        # Domain points: direct copy
                        boundary_mask_s = data.geom.on_boundary(train_x_s)
                        if hasattr(data.geom, "on_initial"):
                            initial_mask_s = data.geom.on_initial(train_x_s)
                        else:
                            initial_mask_s = np.zeros(len(train_x_s), dtype=bool)
                        domain_mask_s = ~(boundary_mask_s | initial_mask_s)

                        mapped_influences[: len(domain_points)] = influences_s[
                            domain_mask_s
                        ]

                        # Boundary points: map to closest boundary point only
                        # Extract only boundary points and their influences from this seed
                        boundary_points_s = train_x_s[boundary_mask_s]
                        boundary_influences_s = influences_s[boundary_mask_s]

                        # For each uniform boundary reference, find closest among boundary points only
                        for i, ref_point in enumerate(uniform_boundary):
                            distances = np.linalg.norm(
                                boundary_points_s - ref_point, axis=1
                            )
                            closest = np.argmin(distances)
                            mapped_influences[len(domain_points) + i] = (
                                boundary_influences_s[closest]
                            )

                        # Initial points: map to closest initial point only
                        if num_initial > 0:
                            # Extract only initial points and their influences from this seed
                            initial_points_s = train_x_s[initial_mask_s]
                            initial_influences_s = influences_s[initial_mask_s]

                            # For each uniform initial reference, find closest among initial points only
                            for i, ref_point in enumerate(uniform_initial):
                                distances = np.linalg.norm(
                                    initial_points_s - ref_point, axis=1
                                )
                                closest = np.argmin(distances)
                                mapped_influences[
                                    len(domain_points) + len(uniform_boundary) + i
                                ] = initial_influences_s[closest]

                        all_influences.append(mapped_influences)
                    else:
                        # When use_test_points=False, influences are indexed by test points
                        # Test points are the same across seeds, so no mapping needed
                        influences_s = scores[:, closest_idx]
                        all_influences.append(influences_s)

            else:
                # Single seed case
                all_influences = []
                train_x = None
                test_x = None
                closest_point = None
                closest_idx = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if train_x is None:
                        train_x = data.train_x_all

                    infl_path = infl_load_path.joinpath(
                        f"{model_name}_influence_scores"
                    )
                    infl_file = infl_path.joinpath(
                        f"influences_{right_term}_{left_term}.npz"
                    )

                    if not infl_file.exists():
                        ax.text(
                            0.5,
                            0.5,
                            f"Missing:\ninfluences_{right_term}_{left_term}.npz",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                            color="red",
                        )
                        ax.set_title(f"{config_name.capitalize()} (seed={s})")
                        continue

                    infl_data = np.load(infl_file)
                    if test_x is None:
                        test_x = infl_data["candidate_points"]
                    scores = infl_data["scores"] * 1 / len(data.train_x_all)

                    if closest_idx is None:
                        if use_test_points:
                            points_to_search = test_x
                        else:
                            points_to_search = train_x

                        distances = np.linalg.norm(
                            points_to_search - target_point, axis=1
                        )
                        closest_idx = np.argmin(distances)
                        closest_point = points_to_search[closest_idx]

                    if use_test_points:
                        influences = scores[closest_idx, :]
                    else:
                        influences = scores[:, closest_idx]

                    all_influences.append(influences)

            if len(all_influences) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No influence files found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"{config_name.capitalize()}")
                continue

            # Average influences across all seeds
            if absolute:
                all_influences = [np.abs(infl) for infl in all_influences]
            influences = np.mean(all_influences, axis=0)

            if alpha is None:
                if absolute:
                    alpha = (influences / influences.max()) ** 2
                else:
                    alpha = 0.7

            coord_scale = 0.1 if problem_name == "navier_stokes_nd" else 1.0

            # Plot influences on training points
            if use_test_points:
                sc = ax.scatter(
                    *(train_x * coord_scale).T,
                    c=influences,
                    cmap="seismic" if not absolute else "Reds",
                    norm=CenteredNorm() if not absolute else None,
                    s=marker_sizes[idx],
                    alpha=alpha,
                )
            else:
                sc = ax.scatter(
                    *(test_x * coord_scale).T,
                    c=influences,
                    cmap="seismic" if not absolute else "Blues",
                    norm=CenteredNorm() if not absolute else None,
                    s=marker_sizes[idx],
                    alpha=alpha,
                )

            # Mark the target/closest point
            ax.scatter(
                *(np.asarray(closest_point) * coord_scale),
                c="black",
                marker="x",
                s=marker_size_cross,
                linewidths=3,
                zorder=10,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(
                    f"{config_name.capitalize()} (avg {len(all_influences)} seeds)"
                )
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

            ax.set_xlabel("x")
            ax.set_ylabel(
                "y" if problem_name in ("poisson_disk", "navier_stokes_nd") else "t"
            )

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Influence")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            ax.set_title(f"{config_name.capitalize()} (seed={seed})")

        # Apply tight layout to each figure
        figures[config_name].tight_layout()

    if use_test_points:
        suffix = "Evaluated on TEST point"
    else:
        suffix = "Evaluated on TRAIN point"

    # Return the two separate figures
    return (figures["good"], axes_dict["good"]), (figures["bad"], axes_dict["bad"])


def visualize_self_influence_comparison(
    problem_name="drift_diffusion",
    seed=0,  # Can be int or "all" to average across all seeds
    figsize=(18, 5),
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    marker_size=30,
):
    """
    Visualize self influences for a target point across different model configurations.

    Parameters:
    -----------
    problem_name : str
        Name of the problem
    seed : int or "all"
        Random seed for model, or "all" to average across all available seeds
    figsize : tuple
        Figure size
    model_zoo_base : str
        Base path to model zoo

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
    }

    fig, axes = plt.subplots(1, len(configs), figsize=figsize, sharey=True)

    for idx, (config_name, (dict_name, params_dict_source)) in enumerate(
        configs.items()
    ):
        ax = axes[idx]

        # Load model
        params_dict = params_dict_source[problem_name]
        load_path = Path(model_zoo_base) / f"{problem_name}_float64"
        infl_load_path = (
            Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"
        )

        # Determine which seeds to use
        if seed == "all":
            if "seeds" in params_dict:
                seeds = params_dict["seeds"]
            else:
                seeds = list(range(10))
        else:
            seeds = [seed]

        try:
            # For averaging across seeds, create uniform reference points
            if seed == "all":
                # Load first model to get geometry and determine number of points
                with Capturing() as cap:
                    model, data, model_name, chkpt_path = load_problem(
                        problem_name=problem_name,
                        params_dict=params_dict,
                        load_path=load_path,
                        seed=seeds[0],
                    )

                # Count boundary and initial points in training set
                train_x_first = data.train_x_all
                boundary_mask = data.geom.on_boundary(train_x_first)
                num_boundary = boundary_mask.sum()

                if hasattr(data.geom, "on_initial"):
                    initial_mask = data.geom.on_initial(train_x_first)
                    num_initial = initial_mask.sum()
                else:
                    num_initial = 0
                    initial_mask = np.zeros(len(train_x_first), dtype=bool)

                # Create uniform reference points for boundary and initial
                uniform_boundary = (
                    data.geom.uniform_boundary_points(num_boundary)
                    if num_boundary > 0
                    else np.empty((0, train_x_first.shape[1]))
                )

                if num_initial > 0 and hasattr(data.geom, "uniform_initial_points"):
                    uniform_initial = data.geom.uniform_initial_points(num_initial)
                else:
                    uniform_initial = np.empty((0, train_x_first.shape[1]))

                # Domain points (non-boundary, non-initial) from first seed
                domain_mask = ~(boundary_mask | initial_mask)
                domain_points = train_x_first[domain_mask]

                # Combine to create reference training set
                train_x = np.vstack([domain_points, uniform_boundary, uniform_initial])

                # Collect influences mapped to reference points
                all_influences = []

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    # Load influence file
                    infl_path = infl_load_path.joinpath(
                        f"{model_name}_influence_scores"
                    )
                    infl_file = infl_path.joinpath(
                        f"influences_total_loss_total_loss_self.npz"
                    )

                    if not infl_file.exists():
                        continue

                    infl_data = np.load(infl_file)
                    influences_s = (
                        infl_data["scores"].diagonal() * 1 / len(data.train_x_all)
                    )
                    train_x_s = data.train_x_all

                    # Map influences to reference points
                    mapped_influences = np.zeros(len(train_x))

                    # Domain points: direct copy (same points across all seeds)
                    mapped_influences[: len(domain_points)] = influences_s[domain_mask]

                    # Boundary points: map to closest boundary point only
                    # Extract only boundary points and their influences from this seed
                    boundary_mask_s = data.geom.on_boundary(train_x_s)
                    boundary_points_s = train_x_s[boundary_mask_s]
                    boundary_influences_s = influences_s[boundary_mask_s]

                    # For each uniform boundary reference, find closest among boundary points only
                    for i, ref_point in enumerate(uniform_boundary):
                        distances = np.linalg.norm(
                            boundary_points_s - ref_point, axis=1
                        )
                        closest_idx = np.argmin(distances)
                        mapped_influences[len(domain_points) + i] = (
                            boundary_influences_s[closest_idx]
                        )

                    # Initial points: map to closest initial point only
                    if num_initial > 0:
                        # Extract only initial points and their influences from this seed
                        initial_mask_s = data.geom.on_initial(train_x_s)
                        initial_points_s = train_x_s[initial_mask_s]
                        initial_influences_s = influences_s[initial_mask_s]

                        # For each uniform initial reference, find closest among initial points only
                        for i, ref_point in enumerate(uniform_initial):
                            distances = np.linalg.norm(
                                initial_points_s - ref_point, axis=1
                            )
                            closest_idx = np.argmin(distances)
                            mapped_influences[
                                len(domain_points) + len(uniform_boundary) + i
                            ] = initial_influences_s[closest_idx]

                    all_influences.append(mapped_influences)

            else:
                # Single seed case
                all_influences = []
                train_x = None

                for s in seeds:
                    with Capturing() as cap:
                        model, data, model_name, chkpt_path = load_problem(
                            problem_name=problem_name,
                            params_dict=params_dict,
                            load_path=load_path,
                            seed=s,
                        )

                    if train_x is None:
                        train_x = data.train_x_all

                    # Load influence file
                    infl_path = infl_load_path.joinpath(
                        f"{model_name}_influence_scores"
                    )
                    infl_file = infl_path.joinpath(
                        f"influences_total_loss_total_loss_self.npz"
                    )

                    if not infl_file.exists():
                        ax.text(
                            0.5,
                            0.5,
                            f"Missing:\ninfluences_total_loss_total_loss_self.npz",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=10,
                            color="red",
                        )
                        ax.set_title(f"{config_name.capitalize()} (seed={s})")
                        continue

                    infl_data = np.load(infl_file)
                    influences_s = (
                        infl_data["scores"].diagonal() * 1 / len(data.train_x_all)
                    )
                    all_influences.append(influences_s)

            if len(all_influences) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No influence files found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                ax.set_title(f"{config_name.capitalize()}")
                continue

            # Average influences across all seeds
            influences = np.mean(all_influences, axis=0)

            sc = ax.scatter(
                *train_x.T,
                c=influences,
                cmap="Reds",
                s=marker_size,
                alpha=0.7,
            )

            # Set title based on whether we're averaging or not
            if seed == "all":
                ax.set_title(
                    f"{config_name.capitalize()} (avg {len(all_influences)} seeds)"
                )
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

            ax.set_xlabel("x")
            if idx == 0:
                ax.set_ylabel("t")

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Influence")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading {config_name}:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            if seed == "all":
                ax.set_title(f"{config_name.capitalize()}")
            else:
                ax.set_title(f"{config_name.capitalize()} (seed={seed})")

    fig.suptitle(f"Self influence (total loss)", fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, axes


def visualize_loss_fractions(
    problem_name="drift_diffusion",
    left_term="output_0",
    seed=0,  # Can be int, list of ints, or "all"
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    figsize=(15, 12),
    cmap="jet",
    markersize=30,
    config_name="good",
    vmin=0.0,
    vmax=1.0,
):
    """
    Visualize the fraction of influence that each loss term contributes to a given test point.

    Since influences are additive, for a fixed left side (e.g., output_0 or total_loss),
    the total influence equals the sum of influences from each right-hand side loss term.
    This function computes and visualizes what fraction each loss term contributes.

    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., 'drift_diffusion', 'burgers')
    left_term : str
        Left side term to analyze (e.g., 'output_0', 'total_loss', 'pde_loss', 'bc_loss')
    seed : int, list, or "all"
        Random seed(s) for model. If "all", averages across all available seeds.
        If list, averages across specified seeds.
    model_zoo_base : str
        Base path to model zoo
    figsize : tuple
        Figure size
    cmap : str
        Colormap for fraction visualization
    markersize : int
        Size of scatter plot markers
    config_name : str
        Configuration to use: "good" or "bad"
    vmin : float
        Minimum value for colorbar (default: 0.0)
    vmax : float
        Maximum value for colorbar (default: 1.0)

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    mean_fractions : dict
        Dictionary containing mean fraction for each loss term
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import Capturing, load_problem

    # Select configuration
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}")

    dict_name, params_dict_source = configs[config_name]
    params_dict = params_dict_source[problem_name]
    load_path = Path(model_zoo_base) / f"{problem_name}_float64"
    infl_load_path = (
        Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"
    )

    # Determine which seeds to use
    if seed == "all":
        if "seeds" in params_dict:
            seeds = params_dict["seeds"]
        else:
            seeds = list(range(10))
    elif isinstance(seed, list):
        seeds = seed
    else:
        seeds = [seed]

    print(f"Using seeds: {seeds}")

    # Load first model to get structure
    with Capturing() as cap:
        model, data, model_name, chkpt_path = load_problem(
            problem_name=problem_name,
            params_dict=params_dict,
            load_path=load_path,
            seed=seeds[0],
        )

    # Get influence scores directory for first seed
    infl_dir = infl_load_path / f"{model_name}_influence_scores"

    # Load a sample file to get num_pdes and num_bcs
    sample_files = list(infl_dir.glob("influences_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No influence files found in {infl_dir}")

    sample_data = np.load(sample_files[0])
    num_pdes = int(sample_data["num_pdes"])
    num_bcs = int(sample_data["num_bcs"])

    print(f"Problem has {num_pdes} PDE term(s) and {num_bcs} BC term(s)")

    # Build list of all right-hand side loss terms
    loss_terms = []

    # Add individual PDE terms
    for pde_idx in range(num_pdes):
        loss_terms.append(f"pde_{pde_idx}")

    # Add individual BC terms
    for bc_idx in range(num_bcs):
        loss_terms.append(f"bc_{bc_idx}")

    print(f"Loss terms to analyze: {loss_terms}")

    # Collect influence scores across all seeds
    all_influences = {term: [] for term in loss_terms}
    test_x = None
    test_mask = None  # Mask for filtering test points based on left_term

    for s in seeds:
        with Capturing() as cap:
            model_s, data_s, model_name_s, chkpt_path_s = load_problem(
                problem_name=problem_name,
                params_dict=params_dict,
                load_path=load_path,
                seed=s,
            )

        infl_dir_s = infl_load_path / f"{model_name_s}_influence_scores"

        for term in loss_terms:
            # Load influence file for this term
            infl_file = infl_dir_s / f"influences_{term}_{left_term}.npz"

            if not infl_file.exists():
                print(f"Warning: {infl_file} not found, skipping")
                continue

            infl_data = np.load(infl_file)

            if test_x is None:
                test_x = infl_data["candidate_points"]

                # Create mask for test points based on left_term
                # If left_term is a BC, only include points on that BC
                if left_term.startswith("bc_"):
                    bc_idx = int(left_term.split("_")[1])
                    if bc_idx < len(model_s.data.bcs):
                        bc = model_s.data.bcs[bc_idx]
                        # Check which test points are on this BC
                        if bc_idx == 0 and hasattr(model_s.data.geom, "on_initial"):
                            test_mask = model_s.data.geom.on_initial(test_x)
                        else:
                            test_mask = model_s.data.bcs[bc_idx].on_boundary(
                                test_x, np.ones(len(test_x), dtype=bool)
                            )
                        print(
                            f"Filtering to {test_mask.sum()} test points on {left_term} (out of {len(test_x)})"
                        )
                    else:
                        raise ValueError(f"BC index {bc_idx} out of range")
                else:
                    # Use all test points
                    test_mask = np.ones(len(test_x), dtype=bool)

            # Extract influence scores
            scores = infl_data["scores"]
            all_influences[term].append(scores)

    if test_x is None:
        raise RuntimeError("Could not load any influence files")

    # Average across seeds for each loss term
    avg_influences = {}
    for term in loss_terms:
        if len(all_influences[term]) > 0:
            avg_influences[term] = np.mean(all_influences[term], axis=0)
        else:
            print(f"Warning: No data found for {term}, skipping")

    if len(avg_influences) == 0:
        raise RuntimeError("No valid influence data found")

    # Apply test mask to filter test points (if BC-specific)
    test_x_filtered = test_x[test_mask]
    avg_influences_filtered = {
        term: scores[test_mask] for term, scores in avg_influences.items()
    }

    # Compute denominator (sum of absolute influences from all terms)
    denominator = np.zeros(test_x_filtered.shape[0])
    for term in avg_influences_filtered.keys():
        denominator += np.abs(avg_influences_filtered[term]).sum(axis=1)

    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-12)

    # Compute fractions for each loss term
    fractions = {}
    mean_fractions = {}

    for term in avg_influences_filtered.keys():
        frac = np.abs(avg_influences_filtered[term]).sum(axis=1) / denominator
        fractions[term] = frac
        mean_fractions[term] = float(frac.mean())

    # Verify fractions sum to 1 (within numerical precision)
    total_fraction = sum(fractions.values())
    print(f"\nFraction sum verification: {total_fraction.mean():.6f} (should be ~1.0)")

    # Print mean fractions
    print("\nMean fractions for each loss term:")
    for term in sorted(mean_fractions.keys()):
        print(f"  {term:15s}: {mean_fractions[term]:.4f}")

    # Create visualization
    n_terms = len(fractions)
    ncols = min(3, n_terms)
    nrows = (n_terms + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot fraction for each loss term
    norm = TwoSlopeNorm(vcenter=0.5, vmin=vmin, vmax=vmax)

    for idx, (term, frac) in enumerate(sorted(fractions.items())):
        ax = axes[idx]

        sc = ax.scatter(
            test_x_filtered[:, 0],
            test_x_filtered[:, 1],
            c=frac,
            cmap=cmap,
            norm=norm,
            s=markersize,
            alpha=0.7,
        )

        ax.set_title(f"{term}\n(mean: {mean_fractions[term]:.3f})", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("t")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Fraction")

    # Hide unused subplots
    for idx in range(n_terms, len(axes)):
        axes[idx].axis("off")

    # Add overall title
    if len(seeds) == 1:
        seed_str = f"seed={seeds[0]}"
    else:
        seed_str = f"avg over {len(seeds)} seeds"

    # Add info about test point filtering
    if left_term.startswith("bc_"):
        filter_str = (
            f" (filtered to {len(test_x_filtered)}/{len(test_x)} points on {left_term})"
        )
    else:
        filter_str = ""

    fig.suptitle(
        f"Loss Fraction Analysis: {left_term} ← loss terms{filter_str}\n{config_name} ({seed_str})",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout()

    return fig, axes, mean_fractions


def visualize_loss_fractions_lineplot(
    problem_name="drift_diffusion",
    left_term="output_0",
    seed=0,  # Can be int, list of ints, or "all"
    model_zoo_base="../../model_zoo_cluster",
    influence_zoo_base=None,
    figsize=(8, 6),
    config_name="good",
    dimension="auto",  # "auto", "time", "space", "radial", or index (0 or 1)
    num_bins=50,  # Number of bins for grouping points along the dimension
    plot_wrt="test",  # "test" or "train" - which points to bin along the dimension
):
    """
    Visualize loss fractions as line plots along a specified dimension.

    For 2D problems (x, t), this averages across one dimension and shows how
    fractions evolve along the other (typically time). For spatial problems,
    can compute radial distance from center.

    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., 'drift_diffusion', 'burgers')
    left_term : str
        Left side term to analyze (e.g., 'output_0', 'total_loss')
    seed : int, list, or "all"
        Random seed(s) for model
    model_zoo_base : str
        Base path to model zoo
    figsize : tuple
        Figure size
    config_name : str
        Configuration to use: "good", "bad", "soap", or "nncg"
    dimension : str or int
        Dimension to plot along:
        - "auto": automatically detect (time for time-dependent, radial for spatial)
        - "time": time dimension (typically index 1)
        - "space": space dimension (typically index 0)
        - "radial": radial distance from center
        - 0 or 1: explicit dimension index
    num_bins : int
        Number of bins for grouping points along the dimension
    plot_wrt : str
        Which points to bin along the dimension:
        - "test": bin test points (default)
        - "train": bin training points

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    results : dict
        Dictionary containing mean fractions, std fractions, mean coherence, and std coherence
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from pinnfluence.utils.defaults import BAD_PROBLEMS, PROBLEMS
    from pinnfluence.utils.utils import Capturing, load_problem

    # Select configuration
    configs = {
        "good": ("PROBLEMS", PROBLEMS),
        "bad": ("BAD_PROBLEMS", BAD_PROBLEMS),
        "soap": ("PROBLEMS", PROBLEMS),
        "nncg": ("PROBLEMS", PROBLEMS),
    }

    if config_name not in configs:
        raise ValueError(f"config_name must be one of {list(configs.keys())}")

    dict_name, params_dict_source = configs[config_name]
    params_dict = dict(params_dict_source[problem_name])
    if config_name == "soap":
        params_dict["optimizer"] = "SOAP"
        params_dict["n_iterations_lbfgs"] = 0
    elif config_name == "nncg":
        params_dict["optimizer"] = "NNCG"
    load_path = Path(model_zoo_base) / f"{problem_name}_float64"
    infl_load_path = (
        Path(influence_zoo_base or model_zoo_base) / f"{problem_name}_float64"
    )

    # Determine which seeds to use
    if seed == "all":
        if "seeds" in params_dict:
            seeds = params_dict["seeds"]
        else:
            seeds = list(range(10))
    elif isinstance(seed, list):
        seeds = seed
    else:
        seeds = [seed]

    print(f"Using seeds: {seeds}")

    # Load first model to get structure
    with Capturing() as cap:
        model, data, model_name, chkpt_path = load_problem(
            problem_name=problem_name,
            params_dict=params_dict,
            load_path=load_path,
            seed=seeds[0],
        )

    # Get influence scores directory for first seed
    infl_dir = infl_load_path / f"{model_name}_influence_scores"

    print(f"Loading influence scores from: {infl_dir}")

    # Load a sample file to get num_pdes and num_bcs
    sample_files = list(infl_dir.glob("influences_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"No influence files found in {infl_dir}")

    sample_data = np.load(sample_files[0])
    num_pdes = int(sample_data["num_pdes"])
    num_bcs = int(sample_data["num_bcs"])

    print(f"Problem has {num_pdes} PDE term(s) and {num_bcs} BC term(s)")

    # Build list of all right-hand side loss terms
    loss_terms = []
    for pde_idx in range(num_pdes):
        loss_terms.append(f"pde_{pde_idx}")
    for bc_idx in range(num_bcs):
        loss_terms.append(f"bc_{bc_idx}")

    print(f"Loss terms to analyze: {loss_terms}")

    # Collect influence scores across all seeds
    all_influences = {term: [] for term in loss_terms}
    test_x = None
    test_mask = None
    train_x = None

    for s in seeds:
        with Capturing() as cap:
            model_s, data_s, model_name_s, chkpt_path_s = load_problem(
                problem_name=problem_name,
                params_dict=params_dict,
                load_path=load_path,
                seed=s,
            )

        infl_dir_s = infl_load_path / f"{model_name_s}_influence_scores"

        for term in loss_terms:
            infl_file = infl_dir_s / f"influences_{term}_{left_term}.npz"

            if not infl_file.exists():
                if term == "pde_0" and problem_name != "navier_stokes_nd":
                    infl_file = infl_dir_s / f"influences_pde_loss_{left_term}.npz"
                    if not infl_file.exists():
                        print(f"Warning: {infl_file} not found, skipping")
                        continue

                elif term == "bc_0" and num_bcs == 1:
                    infl_file = infl_dir_s / f"influences_bc_loss_{left_term}.npz"
                    if not infl_file.exists():
                        print(f"Warning: {infl_file} not found, skipping")
                        continue
                else:
                    print(f"Warning: {infl_file} not found, skipping")
                    continue

            infl_data = np.load(infl_file)

            if test_x is None:
                test_x = infl_data["candidate_points"]
                train_x = data_s.train_x_all

                # Create mask for test points based on left_term
                # If left_term is a BC, only include points on that BC
                if left_term.startswith("bc_"):
                    bc_idx = int(left_term.split("_")[1])
                    if bc_idx < len(model_s.data.bcs):
                        # Check which test points are on this BC
                        if bc_idx == 0 and hasattr(data_s.geom, "on_initial"):
                            test_mask = data_s.geom.on_initial(test_x)
                        else:
                            test_mask = data_s.bcs[bc_idx].on_boundary(
                                test_x, data_s.bcs[bc_idx].geom.on_boundary(test_x)
                            )
                        print(
                            f"Filtering to {test_mask.sum()} test points on {left_term} (out of {len(test_x)})"
                        )
                    else:
                        raise ValueError(f"BC index {bc_idx} out of range")
                else:
                    # Use all test points
                    test_mask = np.ones(len(test_x), dtype=bool)

            scores = infl_data["scores"]
            all_influences[term].append(scores)

    if test_x is None:
        raise RuntimeError("Could not load any influence files")

    if train_x is None:
        raise RuntimeError("Could not load training points")

    # Apply test mask to filter test points (only relevant for test-based plotting)
    if plot_wrt == "test":
        test_x_filtered = test_x[test_mask]
        all_influences_filtered = {
            term: [scores[test_mask] for scores in all_influences[term]]
            for term in loss_terms
        }
        num_points = test_x_filtered.shape[0]
        points_x = test_x_filtered
    else:  # plot_wrt == "train"
        # For training points, we don't filter test points but use all influences
        all_influences_filtered = all_influences
        num_points = train_x.shape[0]
        points_x = train_x

    num_seeds = len(seeds)

    # Compute fractions and pointwise coherence per seed, then average
    all_fractions = {term: [] for term in loss_terms}
    all_coherence = []

    for s_idx in range(num_seeds):
        # Get shape information from first available term
        shape_info = None
        for term in loss_terms:
            if len(all_influences_filtered[term]) > s_idx:
                shape_info = all_influences_filtered[term][s_idx].shape
                break

        if shape_info is None:
            continue

        if plot_wrt == "test":
            # Shape: (num_test_points, n_train)
            num_test_points, n_train = shape_info

            # Compute denominator (sum of absolute influences) for this seed
            # Shape: (num_test_points,) - summed over training points
            denom_s = np.zeros(num_test_points)
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    denom_s += np.abs(all_influences_filtered[term][s_idx]).sum(axis=1)
            denom_s = np.maximum(denom_s, 1e-12)

            # Compute pointwise coherence: for each test point, compute coherence
            # at each training point, then average over training points
            signed_sum_pointwise = np.zeros((num_test_points, n_train))
            abs_sum_pointwise = np.zeros((num_test_points, n_train))

            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    signed_sum_pointwise += all_influences_filtered[term][s_idx]
                    abs_sum_pointwise += np.abs(all_influences_filtered[term][s_idx])

            abs_sum_pointwise = np.maximum(abs_sum_pointwise, 1e-12)

            # Pointwise coherence: |signed| / |abs| at each (test, train) pair
            # Then average over training points for each test point
            pointwise_coherence = np.abs(signed_sum_pointwise) / abs_sum_pointwise
            coherence_s = pointwise_coherence.mean(axis=1)
            all_coherence.append(coherence_s)

            # Compute fractions for this seed
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    frac_s = (
                        np.abs(all_influences_filtered[term][s_idx]).sum(axis=1)
                        / denom_s
                    )
                    all_fractions[term].append(frac_s)

        else:  # plot_wrt == "train"
            # New logic: compute fractions per training point (sum over test points)
            # Shape: (num_test_points, n_train)
            num_test_points, n_train = shape_info

            # Compute denominator (sum of absolute influences) for this seed
            # Shape: (n_train,) - summed over test points
            denom_s = np.zeros(n_train)
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    denom_s += np.abs(all_influences_filtered[term][s_idx]).sum(axis=0)
            denom_s = np.maximum(denom_s, 1e-12)

            # Compute pointwise coherence: for each training point, compute coherence
            # at each test point, then average over test points
            signed_sum_pointwise = np.zeros((num_test_points, n_train))
            abs_sum_pointwise = np.zeros((num_test_points, n_train))

            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    signed_sum_pointwise += all_influences_filtered[term][s_idx]
                    abs_sum_pointwise += np.abs(all_influences_filtered[term][s_idx])

            abs_sum_pointwise = np.maximum(abs_sum_pointwise, 1e-12)

            # Pointwise coherence: |signed| / |abs| at each (test, train) pair
            # Then average over test points for each training point
            pointwise_coherence = np.abs(signed_sum_pointwise) / abs_sum_pointwise
            coherence_s = pointwise_coherence.mean(axis=0)  # shape: (n_train,)
            all_coherence.append(coherence_s)

            # Compute fractions for this seed
            for term in loss_terms:
                if len(all_influences_filtered[term]) > s_idx:
                    frac_s = (
                        np.abs(all_influences_filtered[term][s_idx]).sum(axis=0)
                        / denom_s
                    )
                    all_fractions[term].append(frac_s)

    # Average fractions and coherence across seeds
    fractions = {}
    fractions_std = {}
    mean_fractions = {}
    std_fractions = {}

    for term in loss_terms:
        if len(all_fractions[term]) > 0:
            fractions[term] = np.mean(all_fractions[term], axis=0)
            fractions_std[term] = np.std(all_fractions[term], axis=0)
            mean_fractions[term] = float(fractions[term].mean())
            std_fractions[term] = float(np.mean(fractions_std[term]))
        else:
            print(f"Warning: No data found for {term}, skipping")

    coherence = np.mean(all_coherence, axis=0)
    coherence_std = np.std(all_coherence, axis=0)
    mean_coherence = float(coherence.mean())
    std_coherence = float(np.mean(coherence_std))

    if len(fractions) == 0:
        raise RuntimeError("No valid influence data found")

    # Determine which dimension to use
    if dimension == "auto":
        # Auto-detect: use time for time-dependent problems, radial for spatial
        time_problems = [
            "burgers",
            "diffusion",
            "drift_diffusion",
            "allen_cahn",
            "wave",
        ]
        if problem_name in time_problems:
            dimension = 1  # Time is typically second dimension
            dim_label = "Time"
        elif problem_name == "navier_stokes_nd":
            dimension = 0
            dim_label = "Space"
        else:
            dimension = "radial"
            dim_label = "Radial Distance from Center"
    elif dimension == "time":
        dimension = 1
        dim_label = "Time"
    elif dimension == "space":
        dimension = 0
        dim_label = "Space"
    elif dimension == "radial":
        dim_label = "Radial Distance from Center"
    else:
        # Assume it's an integer index
        dim_label = f"Dimension {dimension}"

    # Compute the coordinate along the chosen dimension
    if dimension == "radial":
        # Distance from center
        center = np.array([0.0, 0.0])  # Disk center at origin
        coord = np.sqrt(((points_x - center) ** 2).sum(axis=1))
    else:
        # Use specified dimension
        coord = points_x[:, dimension]

    # Sort by coordinate for cleaner line plots
    sort_idx = np.argsort(coord)
    coord_sorted = coord[sort_idx]
    fractions_sorted = {term: frac[sort_idx] for term, frac in fractions.items()}
    fractions_std_sorted = {term: std[sort_idx] for term, std in fractions_std.items()}
    coherence_sorted = coherence[sort_idx]
    coherence_std_sorted = coherence_std[sort_idx]

    # Bin the data to reduce noise
    bins = np.linspace(coord_sorted.min(), coord_sorted.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_fractions = {term: np.zeros(num_bins) for term in fractions.keys()}
    binned_fractions_std = {term: np.zeros(num_bins) for term in fractions.keys()}
    binned_coherence = np.zeros(num_bins)
    binned_coherence_std = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (coord_sorted >= bins[i]) & (coord_sorted < bins[i + 1])
        if mask.sum() > 0:
            for term in fractions.keys():
                binned_fractions[term][i] = fractions_sorted[term][mask].mean()
                binned_fractions_std[term][i] = fractions_std_sorted[term][mask].mean()
            binned_coherence[i] = coherence_sorted[mask].mean()
            binned_coherence_std[i] = coherence_std_sorted[mask].mean()

    if problem_name == "navier_stokes_nd":
        bin_centers = bin_centers * 0.1

    # Create line plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the cancellation factor (1 - coherence)
    ax.plot(
        bin_centers,
        1 - binned_coherence,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"Cancellation $\\kappa$ (mean: {1 - mean_coherence:.3f})",
    )

    # # Plot each loss term
    for term in sorted(binned_fractions.keys()):
        ax.plot(
            bin_centers,
            binned_fractions[term],
            label=f"{loss_term_names[problem_name][term]} (mean: {mean_fractions[term]:.3f})",
            linewidth=2,
        )
        ax.fill_between(
            bin_centers,
            binned_fractions[term] - binned_fractions_std[term],
            binned_fractions[term] + binned_fractions_std[term],
            alpha=0.2,
        )

    ax.set_xlabel(dim_label)
    ax.set_ylabel("Loss-Fraction")
    ax.set_ylim(0, 1)
    # ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add overall title
    if len(seeds) == 1:
        seed_str = f"seed={seeds[0]}"
    else:
        seed_str = f"avg over {len(seeds)} seeds"

    if plot_wrt == "test" and left_term.startswith("bc_"):
        filter_str = f" (filtered to {len(test_x_filtered)}/{len(test_x)} test points on {left_term})"
    else:
        filter_str = ""

    points_type = "test points" if plot_wrt == "test" else "training points"
    ax.set_title(
        f"Loss Fraction vs {dim_label} ({points_type}): {loss_term_names[problem_name][left_term]} ← loss terms{filter_str}\n{config_name} ({seed_str})",
        fontsize=14,
    )

    plt.tight_layout()

    # Compile results
    results = {
        "mean_fractions": mean_fractions,
        "std_fractions": std_fractions,
        "mean_coherence": mean_coherence,
        "std_coherence": std_coherence,
        "binned_fractions": binned_fractions,
        "binned_fractions_std": binned_fractions_std,
        "binned_coherence": binned_coherence,
        "bin_centers": bin_centers,
    }

    return fig, ax, results


def create_horizontal_legend(
    colors,
    labels,
    linestyles=None,
    linewidths=None,
    markers=None,
    figsize=(16, 0.5),
    ncol=None,
    frameon=False,
    loc="center",
    save_path=None,
    dpi=300,
):
    """
    Create a standalone horizontal legend figure.

    Parameters:
    -----------
    colors : list
        List of colors for each legend entry
    labels : list
        List of labels for each legend entry
    linestyles : list, optional
        List of linestyles (e.g., '-', '--', '-.', ':'). If None, uses solid lines.
    linewidths : list or float, optional
        List of linewidths or single value. Default is 2.
    markers : list, optional
        List of markers (e.g., 'o', 's', '^'). If None, no markers.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 0.5).
    fontsize : int, optional
        Font size for legend text. Default is 12.
    ncol : int, optional
        Number of columns. If None, uses len(labels).
    frameon : bool, optional
        Whether to draw frame around legend. Default is False.
    loc : str, optional
        Location of legend. Default is 'center'.
    save_path : str, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        DPI for saved figure. Default is 300.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects

    Examples:
    ---------
    # Simple line legend
    create_horizontal_legend(
        colors=['red', 'blue', 'green'],
        labels=['Model A', 'Model B', 'Model C'],
        save_path='legend.pdf'
    )

    # With different linestyles
    create_horizontal_legend(
        colors=['red', 'blue', 'green'],
        labels=['Train', 'Val', 'Test'],
        linestyles=['-', '--', '-.'],
        linewidths=3
    )

    # With markers
    create_horizontal_legend(
        colors=['red', 'blue'],
        labels=['Method 1', 'Method 2'],
        markers=['o', 's'],
        linestyles=['-', '-']
    )
    """
    # Set defaults
    if linestyles is None:
        linestyles = ["-"] * len(labels)
    if linewidths is None:
        linewidths = [2] * len(labels)
    elif isinstance(linewidths, (int, float)):
        linewidths = [linewidths] * len(labels)
    if markers is None:
        markers = [None] * len(labels)
    if ncol is None:
        ncol = len(labels)

    # Create legend handles
    handles = []
    for color, label, ls, lw, marker in zip(
        colors, labels, linestyles, linewidths, markers
    ):
        handle = Line2D(
            [0],
            [0],
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker=marker,
            markersize=8,
            label=label,
        )
        handles.append(handle)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Create legend
    legend = ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=frameon)

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    return fig, ax

