"""
Pretrain a PINN model on a given problem with Adam (+ optional L-BFGS).

To generate a randomly initialized (untrained) model, use --n_iterations=0.

Usage:
    python -m pinnfluence.pretrain [options]

    Use --help to see all available options.

Example:
    python -m pinnfluence.pretrain --problem burgers --n_iterations 15_000 --num_domain 1_000
    python -m pinnfluence.pretrain --problem burgers --n_iterations 15_000 --num_domain 1_000 --n_iterations_lbfgs 5_000 --float64
"""

from functools import partial
from pathlib import Path

import deepxde as dde

from . import problem_factory
from .utils.callbacks import (BestModelCheckpoint, EvalMetricCallback,
                              WandbCallbackLoss, WandbCallbackPlots)
from .utils.defaults import DEFAULTS
from .utils.parse_args import parse_pretrain_args as parse_args
from .utils.utils import plot_prediction_heatmap, set_default_device


def main(
    # see defaults.py for default values when called from command line
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
    device: str = "cpu",
    soft_constrained: bool = True,
    load_path: str = None,
    wandb: bool = False,
    drop_single_point_type: str = "none",
) -> None:
    set_default_device(device)

    if soft_constrained:
        print("SOFT CONSTRAINING")

    if use_float64:
        dde.config.set_default_float("float64")
    dde.config.set_random_seed(seed)

    # Construct the problem and load the pretrained checkpoint
    model, data, model_name, _ = problem_factory.construct_problem(
        problem_name=problem_name,
        lr=lr,
        layers=layers,
        n_iterations=n_iterations,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        optimizer=optimizer,
        seed=seed,
        float64=use_float64,
        force_reinit=True,  # don't continue training
        soft_constrained=soft_constrained,
        load_path=load_path,
        drop_single_point_type=drop_single_point_type,
    )

    if wandb:
        cmap = "jet" if problem_name.startswith("navier_stokes") else "coolwarm"
        wandb_callback_plots = WandbCallbackPlots(
            period=1000,
            project="thesis",
            name=model_name,
            plotting_func=partial(plot_prediction_heatmap, cmap=cmap),
        )
        wandb_callback_loss = WandbCallbackLoss(
            period=100, project="thesis", name=model_name
        )
        wandb_callbacks = [wandb_callback_plots, wandb_callback_loss]
    else:
        wandb_callbacks = []

    Path(save_path).mkdir(parents=True, exist_ok=True)

    model.train(
        n_iterations,
        display_every=100,
        verbose=0,
        callbacks=wandb_callbacks
        + [
            # store final model
            BestModelCheckpoint(
                filepath=f"{save_path}/{model_name}_full.pt",
                verbose=False,
                save_better_only=False,
                period=1000,
            ),
            # store metrics throughout training
            EvalMetricCallback(
                filepath=f"{save_path}/{model_name}_eval.csv",
                verbose=True,
                total_epochs=n_iterations,
                verbose_period=5000,
            ),
        ],
    )

    if n_iterations_lbfgs > 0:
        dde.optimizers.config.set_LBFGS_options(maxiter=n_iterations_lbfgs)
        model_name = model_name.replace("0_lbfgs", f"{n_iterations_lbfgs}_lbfgs")
        model = problem_factory.compile_model(
            net=model.net,
            data=data,
            lr=lr,
            optimizer="L-BFGS",
        )

        model.train(
            n_iterations,
            display_every=100,
            verbose=0,
            callbacks=wandb_callbacks
            + [
                # store best checkpoints
                BestModelCheckpoint(
                    filepath=f"{save_path}/{model_name}_full.pt",
                    verbose=False,
                    save_better_only=False,
                ),
                # store metrics throughout training
                EvalMetricCallback(
                    filepath=f"{save_path}/{model_name}_eval.csv",
                    verbose=1,
                    total_epochs=n_iterations_lbfgs,
                    verbose_period=5000,
                ),
            ],
        )


if __name__ == "__main__":
    args = parse_args()
    main(
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
        device=args.device,
        soft_constrained=args.soft_constrained,
        load_path=args.load_path,
        wandb=args.wandb,
        drop_single_point_type=args.drop_single_point_type,
    )
