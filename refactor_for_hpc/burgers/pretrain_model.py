import argparse
import numpy as np
import deepxde as dde
import os
import torch

from defaults import DEFAULTS
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.callbacks import BestModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"], help="Random seed"
    )
    parser.add_argument(
        "--lr",
        type=int,
        nargs="+",
        default=DEFAULTS["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULTS["layers"],
        help="Layer sizes",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=DEFAULTS["n_iterations"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--n_iterations_lbfgs",
        type=int,
        default=DEFAULTS["n_iterations_lbfgs"],
        help="Number of iterations for L-BFGS",
    )
    parser.add_argument(
        "--num_domain",
        type=int,
        default=DEFAULTS["num_domain"],
        help="Number of domain points",
    )
    parser.add_argument(
        "--num_boundary",
        type=int,
        default=DEFAULTS["num_boundary"],
        help="Number of boundary points",
    )
    parser.add_argument(
        "--num_initial",
        type=int,
        default=DEFAULTS["num_initial"],
        help="Number of initial points",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULTS["save_path"],
        help="Path to save model",
    )
    parser.add_argument(
        "--hard_constrained", action="store_true", help="Use hard constrained model"
    )
    parser.add_argument(
        "--train_distribution",
        type=str,
        default=DEFAULTS["train_distribution"],
        help="Distribution of training points",
    )
    return parser.parse_args()


def burgers_equation(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / dde.backend.torch.pi * dy_xx


# define the geometry and time domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# define the boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: -dde.backend.torch.sin(
        dde.backend.torch.pi * dde.backend.as_tensor(x[:, 0:1])
    ),
    lambda _, on_initial: on_initial,
)


def main(args):
    lr = args.lr
    layers = args.layers
    n_iterations = args.n_iterations
    num_domain = args.num_domain
    num_boundary = args.num_boundary
    num_initial = args.num_initial
    save_path = args.save_path
    seed = args.seed
    hard_constrained = args.hard_constrained
    n_iterations_lbfgs = args.n_iterations_lbfgs
    train_distribution = args.train_distribution

    dde.config.set_default_float("float64")

    dde.config.set_random_seed(seed)
    dde.optimizers.config.set_LBFGS_options(maxiter=n_iterations_lbfgs)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    net = dde.maps.FNN(layers, "tanh", "Glorot normal")

    if hard_constrained:

        def output_transform(x, y):
            return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

        net.apply_output_transform(output_transform)
        ic_bcs = []

        model_name = (
            f"{save_path}/adam_{n_iterations}_best_seed_{seed}_hardconstrained.pt"
        )

    else:
        ic_bcs = [bc, ic]

        model_name = f"{save_path}/adam_{n_iterations}_best_seed_{seed}.pt"

    if layers != DEFAULTS["layers"]:
        model_name = model_name.replace(".pt", f"_layers_{layers}.pt")

    if train_distribution != DEFAULTS["train_distribution"]:
        model_name = model_name.replace(
            ".pt", f"_train_distribution_{train_distribution}.pt"
        )

    if n_iterations_lbfgs > 0:
        model_name = model_name.replace(
            f"adam_{n_iterations}", f"adam_{n_iterations}_lbfgs_{n_iterations_lbfgs}"
        )

    data = dde.data.TimePDE(
        geomtime,
        burgers_equation,
        ic_bcs=ic_bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=10_000,
        train_distribution=train_distribution,
    )

    data_filename = (
        f"{save_path}/data_{num_domain}_{num_boundary}_{num_initial}_{seed}.npz"
    )

    if train_distribution != DEFAULTS["train_distribution"]:
        data_filename = data_filename.replace(
            ".npz", f"_train_distribution_{train_distribution}.npz"
        )

    np.savez_compressed(
        data_filename,
        train_x_all=data.train_x_all,
        train_x=data.train_x,
        train_x_bc=data.train_x_bc,
        test_x=data.test_x,
    )

    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    model.train(
        iterations=n_iterations,
        display_every=1000,
        callbacks=[BestModelCheckpoint(model_name, verbose=0, save_better_only=False)],
    )

    if n_iterations_lbfgs > 0:
        model.compile("L-BFGS")
        model.train(
            iterations=n_iterations_lbfgs,
            display_every=1000,
            callbacks=[
                BestModelCheckpoint(model_name, verbose=0, save_better_only=False)
            ],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
