import argparse
import os
import numpy as np
import pandas as pd
import torch
import deepxde as dde
from pathlib import Path

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.callbacks import BestModelCheckpoint
from utils.utils import sample_new_training_points_via_IF
from pretrain_model import burgers_equation, ic, bc, geomtime
from defaults import DEFAULTS


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
        "--method",
        type=str,
        default="IF",
        help="Method to evaluate",
        choices=["IF", "random", "RAR", "control"],
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        default=10_000,
        help="Number of candidate points to sample from",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Fraction of training points to add (from sampled ones)",
    )
    parser.add_argument(
        "--influence_sign",
        type=str,
        default="abs",
        help="Sign of influence scores",
        choices=["abs", "pos", "neg"],
    )
    return parser.parse_args()


def gen_testdata():
    data = np.load("/opt/code/burgers/dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def eval_model(model, x, y, verbose=True):
    y_pred = model.predict(x)
    f = model.predict(x, operator=burgers_equation)
    mean_residual = np.mean(np.abs(f))
    l2_relative_error = dde.metrics.l2_relative_error(y, y_pred)
    mse = dde.metrics.mean_squared_error(y, y_pred)
    if verbose:
        print(f"Mean residual: {mean_residual:.2e}")
        print(f"L2 relative error: {l2_relative_error:.2e}")
        print(f"MSE: {mse:.2e}")
    return {
        "mean_residual": mean_residual,
        "l2_relative_error": l2_relative_error,
        "mse": mse,
    }


def reset_data(data, data_points):
    data.train_x_all = data_points["train_x_all"]
    data.train_x = data_points["train_x"]
    data.train_x_bc = data_points["train_x_bc"]
    data.test_x = data_points["test_x"]
    return data


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
    n_candidate_points = args.n_candidate_points
    ratio = args.ratio
    method = args.method
    influence_sign = args.influence_sign

    dde.config.set_random_seed(seed)

    X_test, y_true = gen_testdata()

    net = dde.maps.FNN(layers, "tanh", "Glorot normal")

    if hard_constrained:

        def output_transform(x, y):
            return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

        net.apply_output_transform(output_transform)
        ic_bcs = []

        model_name_src = f"/opt/model_zoo_src/adam_{n_iterations}_best_seed_{seed}_hardconstrained.pt"
    else:
        ic_bcs = [bc, ic]

        model_name_src = f"/opt/model_zoo_src/adam_{n_iterations}_best_seed_{seed}.pt"

    net.load_state_dict(torch.load(model_name_src)["model_state_dict"])

    data = dde.data.TimePDE(
        geomtime,
        burgers_equation,
        ic_bcs=ic_bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=10_000,
    )

    data_points = np.load(
        f"/opt/model_zoo_src/data_{num_domain}_{num_boundary}_{num_initial}_{seed}.npz"
    )
    reset_data(data, data_points)

    if method == "IF":
        # note: this only affects PDE loss for now
        influence_scores = np.load(
            f"/opt/model_zoo_src/influence_scores_{n_iterations}_{num_domain}_{num_boundary}_{num_initial}_{seed}_hc_{hard_constrained}.npz"
        )
        infl_scores = influence_scores["influence_scores"]
        candidate_points = influence_scores["candidate_points"]
        if influence_sign == "abs":
            summed_infl_scores = np.abs(infl_scores).sum(axis=0)
        elif influence_sign == "pos":
            summed_infl_scores = infl_scores.sum(axis=0)
        else:
            summed_infl_scores = (-infl_scores).sum(axis=0)
        topk_idx = torch.topk(
            torch.tensor(summed_infl_scores),
            int(ratio * len(data.train_x_all)),
            dim=0,
        )[1].numpy()
        new_anchors = candidate_points[topk_idx]
        data.add_anchors(new_anchors)

    elif method == "random":
        new_anchors = geomtime.random_points(int(ratio * len(data.train_x_all)))
        data.add_anchors(new_anchors)

    elif method == "RAR":
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)

        candidate_points = geomtime.random_points(n_candidate_points)
        residual = np.abs(model.predict(candidate_points, operator=burgers_equation))[
            :, 0
        ]
        err_eq = torch.tensor(residual)
        topk_idx = torch.topk(err_eq, int(ratio * len(data.train_x_all)), dim=0)[
            1
        ].numpy()
        new_anchors = candidate_points[topk_idx]
        data.add_anchors(new_anchors)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    model.train(
        epochs=10_000,
        callbacks=[
            BestModelCheckpoint(
                f"{save_path}/adam_{n_iterations}_seed_{seed}_{method}_train_{influence_sign}.pt",
                verbose=1,
                save_better_only=True,
                monitor="train loss",
            ),
            BestModelCheckpoint(
                f"{save_path}/adam_{n_iterations}_seed_{seed}_{method}_test_{influence_sign}.pt",
                verbose=1,
                save_better_only=True,
                monitor="test loss",
            ),
            BestModelCheckpoint(
                f"{save_path}/adam_{n_iterations}_seed_{seed}_{method}_10k_{influence_sign}.pt",
                save_better_only=False,
            ),
        ],
    )

    df = pd.DataFrame()

    for model_name in ["train", "test", "10k"]:
        model.net.load_state_dict(
            torch.load(
                f"{save_path}/adam_{n_iterations}_seed_{seed}_{method}_{model_name}_{influence_sign}.pt",
            )["model_state_dict"]
        )

        metrics = eval_model(model, X_test, y_true)

        metrics["method"] = method
        metrics["ratio"] = ratio
        metrics["seed"] = seed
        metrics["n_iterations"] = n_iterations
        metrics["num_domain"] = num_domain
        metrics["num_boundary"] = num_boundary
        metrics["num_initial"] = num_initial
        metrics["n_candidate_points"] = n_candidate_points
        metrics["hard_constrained"] = hard_constrained
        metrics["model_version"] = model_name
        metrics["influence_sign"] = influence_sign

        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    df.to_csv(
        f"{save_path}/metrics_ratio_{ratio:.3f}_{method}_{seed}_hc_{hard_constrained}_{influence_sign}.csv",
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
