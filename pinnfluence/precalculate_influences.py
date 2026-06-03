"""
This script precalculates the influence scores for a given model and saves them to a .npz file.

This is useful for evaluation of scoring and sampling strategies without the need to recalculate the influence scores
for each experiment.

Usage:
    python -m pinnfluence.precalculate_influences [options]

    Use --help to see all available options.
"""

import os

import deepxde as dde
import numpy as np

from . import problem_factory
from .utils.models import ModelWrapper, NetPredWrapper, PINNLoss
from .utils.parse_args import parse_precalculate_args as parse_args
from .utils.influence import (calculate_influence_scores, instantiate_grad_dot,
                              instantiate_IF, sample_random_points)
from .utils.utils import set_default_device


def parse_loss_term(term_str, model, num_pdes, num_bcs):
    """
    Parses a loss term string and returns configuration for ModelWrapper/NetPredWrapper and PINNLoss.

    Args:
        term_str: String like "total_loss", "pde_loss", "bc_loss", "pde_0", "bc_1", "output_0"
        model: The PINN model object
        num_pdes: Number of PDE terms
        num_bcs: Number of boundary condition terms

    Returns:
        Tuple of (model_wrapper, loss_fn, test_reduction_type)
    """
    if term_str == "total_loss":
        model_wrapper = ModelWrapper(
            net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=True
        )
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str == "pde_loss":
        model_wrapper = ModelWrapper(
            net=model.net, pde=model.data.pde, bcs=[], include_pde=True
        )
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str == "bc_loss":
        model_wrapper = ModelWrapper(
            net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
        )
        loss_fn = PINNLoss(include_all_losses=True)
        test_reduction_type = "none"
    elif term_str.startswith("pde_"):
        try:
            pde_idx = int(term_str.split("_")[1])
            if pde_idx >= num_pdes:
                raise ValueError(f"PDE index {pde_idx} out of range (0-{num_pdes - 1})")
            model_wrapper = ModelWrapper(
                net=model.net, pde=model.data.pde, bcs=[], include_pde=True
            )
            loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[pde_idx])
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid PDE term format: {term_str}. Expected 'pde_N' where N is an integer"
            ) from e
    elif term_str.startswith("bc_"):
        try:
            bc_idx = int(term_str.split("_")[1])
            if bc_idx >= num_bcs:
                raise ValueError(f"BC index {bc_idx} out of range (0-{num_bcs - 1})")
            model_wrapper = ModelWrapper(
                net=model.net, pde=model.data.pde, bcs=model.data.bcs, include_pde=False
            )
            loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[bc_idx])
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid BC term format: {term_str}. Expected 'bc_N' where N is an integer"
            ) from e
    elif term_str.startswith("output_"):
        try:
            output_idx = int(term_str.split("_")[1])
            n_outputs = model.net.linears[-1].out_features
            if output_idx >= n_outputs:
                raise ValueError(
                    f"Output index {output_idx} out of range (0-{n_outputs - 1})"
                )
            model_wrapper = NetPredWrapper(model.net, pred_idx=output_idx)
            loss_fn = None
            test_reduction_type = "none"
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid output term format: {term_str}. Expected 'output_N' where N is an integer"
            ) from e
    else:
        raise ValueError(
            f"Unknown loss term: {term_str}. Valid options: total_loss, pde_loss, bc_loss, pde_N, bc_N, output_N"
        )

    return model_wrapper, loss_fn, test_reduction_type


def compute_single_influence(
    IF_instance,
    left_term,
    right_term,
    model,
    candidate_points,
    batch_size,
    num_pdes,
    num_bcs,
):
    """
    Computes a single influence matrix for the given left/right configuration.

    Args:
        IF_instance: The instantiated influence function object
        left_term: String describing the test loss term (left side)
        right_term: String describing the training loss term (right side)
        model: The PINN model object
        candidate_points: The candidate points to compute influence for
        batch_size: Batch size for computation
        num_pdes: Number of PDE terms
        num_bcs: Number of BC terms

    Returns:
        Influence scores as numpy array
    """
    # Configure right side (training loss)
    right_model, right_loss, _ = parse_loss_term(right_term, model, num_pdes, num_bcs)
    IF_instance.model = right_model
    IF_instance.loss_fn = right_loss

    # Configure left side (test loss)
    left_model, left_loss, left_reduction = parse_loss_term(
        left_term, model, num_pdes, num_bcs
    )
    IF_instance.model_test = left_model
    IF_instance.test_loss_fn = left_loss
    IF_instance.test_reduction_type = left_reduction

    print(f"Computing influence: right={right_term}, left={left_term}")
    infl_scores = calculate_influence_scores(
        tda_instance=IF_instance,
        candidate_points=candidate_points,
        batch_size=batch_size,
        show_progress=True,
    ).astype(np.float32)

    return infl_scores


def main(
    n_candidate_points: int = 10_000,
    seed: int = 42,
    lr: float = 0.001,
    layers: list = [2] + [32] * 3 + [1],
    n_iterations: int = 10_000,
    n_iterations_lbfgs: int = 0,
    num_domain: int = 1_000,
    num_boundary: int = 0,
    num_initial: int = 0,
    save_path: str = "./model_zoo",
    problem_name: str = "burgers",
    optimizer: str = "adam",
    use_float64: bool = False,
    scoring_method: str = "PINNfluence",
    use_holdout_test: bool = False,
    precalc_infl_sample_uniformly: bool = False,
    device: str = "cpu",
    soft_constrained: bool = True,
    use_train_set: bool = False,
    load_path: str = None,
    left: str = None,
    right: str = None,
    self_influence: bool = False,
):
    # Validate arguments
    if left is None or right is None:
        raise ValueError("Both --left and --right must be specified")

    assert scoring_method in [
        "PINNfluence",
        "grad_dot",
    ], "Can only precompute PINNfluence or grad_dot"
    if use_float64:
        dde.config.set_default_float("float64")
    dde.config.set_random_seed(seed)

    set_default_device(device)

    # Construct the problem and load the pretrained checkpoint
    model, data, model_name, chkpt_path = problem_factory.construct_problem(
        problem_name=problem_name,
        lr=lr,
        layers=layers,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        optimizer=optimizer,
        seed=seed,
        float64=use_float64,
        soft_constrained=soft_constrained,
        load_path=load_path,
    )

    assert chkpt_path is not None, (
        "Could not load checkpoint. Influences shall be only calculated for already trained models"
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Single influence matrix computation
    # Get number of PDEs and BCs
    n_outputs = model.net.linears[-1].out_features
    num_pdes = n_outputs
    num_bcs = len(model.data.bcs)

    # Setup candidate points
    dde.config.set_random_seed(42)

    if self_influence:
        candidate_points = model.data.train_x_all
        print(
            f"Using self-influence mode: {len(candidate_points)} training points as candidates"
        )
    elif use_holdout_test:
        candidate_points = model.data.holdout_test_x
        print(f"Using holdout test set: {len(candidate_points)} points as candidates")
    else:
        if precalc_infl_sample_uniformly:
            candidate_points = data.geom.uniform_points(n_candidate_points)
        else:
            candidate_points = sample_random_points(
                geometry=model.data.geom,
                num_points=n_candidate_points,
                num_bcs=num_bcs,
            )
        print(f"Sampled {len(candidate_points)} candidate points")

    batch_size = 1024

    # Instantiate IF
    print("Instantiating influence function...")
    if scoring_method == "grad_dot":
        IF_instance = instantiate_grad_dot(
            model, use_train_set=True, show_progress=False
        )
    else:
        IF_instance = instantiate_IF(
            model,
            use_train_set=True,
            show_progress=False,
            model_name=model_name,
            prefer_load_R=False,
        )

    # Compute single influence matrix
    infl_scores = compute_single_influence(
        IF_instance=IF_instance,
        left_term=left,
        right_term=right,
        model=model,
        candidate_points=candidate_points,
        batch_size=batch_size,
        num_pdes=num_pdes,
        num_bcs=num_bcs,
    )

    print(f"Computed influence scores with shape: {infl_scores.shape}")

    # Create subdirectory for saving
    influence_dir = os.path.join(save_path, f"{model_name}_influence_scores")
    if not os.path.exists(influence_dir):
        os.makedirs(influence_dir)

    # Create filename
    suffix = "_self" if self_influence else ""
    prefix = "influences" if scoring_method == "PINNfluence" else "grad_dot"
    filename = f"{prefix}_{right}_{left}{suffix}.npz"
    filepath = os.path.join(influence_dir, filename)

    # Save to file
    np.savez_compressed(
        filepath,
        scores=infl_scores,
        candidate_points=candidate_points,
        num_pdes=num_pdes,
        num_bcs=num_bcs,
        n_outputs=n_outputs,
        left_term=left,
        right_term=right,
        self_influence=self_influence,
    )
    print(f"\nSaved influence matrix to {filepath}")


if __name__ == "__main__":
    args = parse_args()
    main(
        n_candidate_points=args.n_candidate_points,
        seed=args.seed,
        lr=args.lr,
        layers=args.layers,
        n_iterations=args.n_iterations,
        n_iterations_lbfgs=args.n_iterations_lbfgs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        save_path=args.save_path,
        problem_name=args.problem,
        optimizer=args.optimizer,
        use_float64=args.float64,
        scoring_method=args.scoring_method,
        use_holdout_test=args.precalc_infl_use_holdout_test,
        precalc_infl_sample_uniformly=args.precalc_infl_sample_uniformly,
        device=args.device,
        soft_constrained=args.soft_constrained,
        use_train_set=True,
        load_path=args.load_path,
        left=args.left,
        right=args.right,
        self_influence=getattr(args, "self_influence", False),
    )
