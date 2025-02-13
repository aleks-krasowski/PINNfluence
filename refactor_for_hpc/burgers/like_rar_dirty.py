import argparse
import os
import numpy as np
import pandas as pd
import torch
import deepxde as dde
from pathlib import Path
from functools import partial

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.utils import sample_new_training_points_via_IF
from pretrain_model import burgers_equation, ic, bc, geomtime
from eval_finetuning import gen_testdata, eval_model, reset_data
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
        choices=["IF", "random", "RAR"],
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        default=10_000,
        help="Number of candidate points to sample from",
    )
    parser.add_argument(
        "--n_sample_points",
        type=int,
        default=10,
        help="Number of points to sample in each iteration",
    )
    parser.add_argument(
        "--distribution_based",
        action="store_true",
        help="Use distribution based sampling",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULTS["k"],
        help="Exponent of distribution based sampling",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=DEFAULTS["c"],
        help="Constant of distribution based sampling",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace training points with new points",
    )
    parser.add_argument(
        "--train_distribution",
        type=str,
        default=DEFAULTS["train_distribution"],
        help="Distribution of training points",
    )
    parser.add_argument(
        "--n_iterations_lbfgs",
        type=int,
        default=DEFAULTS["n_iterations_lbfgs"],
        help="Number of iterations for L-BFGS",
    )
    parser.add_argument(
        "--use_float64", action="store_true", help="Use float64 for training"
    )
    return parser.parse_args()


def sample_n_from_candidates(
    candidate_points,
    scores,
    num_points,
):
    scores = torch.tensor(scores)
    topk_idx = torch.topk(scores, num_points, dim=0)[1].numpy()
    sample = candidate_points[topk_idx]

    return sample


def sample_from_distribution(candidate_points, scores, num_points, k=1, c=1):
    scores = np.power(scores, k) / np.power(scores, k).mean() + c
    scores = scores / scores.sum()
    sample = np.random.choice(
        a=len(candidate_points), size=num_points, p=scores, replace=False
    )

    return candidate_points[sample]


def sample_random_points(_, __, num_points):
    return geomtime.random_points(num_points)


def get_influence_scores(
    model,
    data,
    n_candidate_points,
):
    influences, candidate_points, _ = sample_new_training_points_via_IF(
        model.net,
        data,
        fraction_of_train=n_candidate_points / len(data.train_x_all),
    )
    influences = np.abs(influences).sum(axis=0)
    return influences, candidate_points


def get_residual_scores(
    model,
    data,
    n_candidate_points,
):
    candidate_points = geomtime.random_points(n_candidate_points)
    residual = np.abs(model.predict(candidate_points, operator=burgers_equation))[:, 0]
    return residual, candidate_points


# dummy function
def get_random_scores(bli, bla, blu):
    return None, None


def main(args):
    layers = args.layers
    n_iterations = args.n_iterations
    n_iterations_lbfgs = args.n_iterations_lbfgs
    num_domain = args.num_domain
    num_boundary = args.num_boundary
    num_initial = args.num_initial
    save_path = args.save_path
    seed = args.seed
    hard_constrained = args.hard_constrained
    n_candidate_points = args.n_candidate_points
    n_sample_points = args.n_sample_points
    method = args.method
    distribution_based = args.distribution_based
    k = args.k
    c = args.c
    replace = args.replace
    train_distribution = args.train_distribution
    use_float64 = args.use_float64

    if use_float64:
        dde.config.set_default_float("float64")

    dde.config.set_random_seed(seed)
    dde.optimizers.config.set_LBFGS_options(maxiter=1000)

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

    if layers != DEFAULTS["layers"]:
        model_name_src = model_name_src.replace(".pt", f"_layers_{layers}.pt")

    if train_distribution != DEFAULTS["train_distribution"]:
        model_name_src = model_name_src.replace(
            ".pt", f"_train_distribution_{train_distribution}.pt"
        )

    if n_iterations_lbfgs > 0:
        model_name_src = model_name_src.replace(
            f"adam_{n_iterations}", f"adam_{n_iterations}_lbfgs_{n_iterations_lbfgs}"
        )

    # net.load_state_dict(torch.load(model_name_src)["model_state_dict"])

    data = dde.data.TimePDE(
        geomtime,
        burgers_equation,
        ic_bcs=ic_bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=10_000,
    )

    # data_filename = (
    #     f"/opt/model_zoo_src/data_{num_domain}_{num_boundary}_{num_initial}_{seed}.npz"
    # )

    # if train_distribution != DEFAULTS["train_distribution"]:
    #     data_filename = data_filename.replace(
    #         ".npz", f"_train_distribution_{train_distribution}.npz"
    #     )

    # data_points = np.load(data_filename)

    # if use_float64:
    #     data_points["train_x_all"] = data_points["train_x_all"].astype(np.float64)
    #     data_points["train_x"] = data_points["train_x"].astype(np.float64)
    #     data_points["train_x_bc"] = data_points["train_x_bc"].astype(np.float64)
    #     data_points["test_x"] = data_points["test_x"].astype(np.float64)

    # reset_data(data, data_points)

    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr)

    model.train(iterations=n_iterations, display_every=100)

    if n_iterations_lbfgs > 0:
        model.compile("L-BFGS")
        model.train()

    X_test, y_true = gen_testdata()

    metrics = eval_model(model, X_test, y_true)

    metrics["method"] = method
    metrics["seed"] = seed
    metrics["n_iterations"] = n_iterations
    metrics["num_domain"] = num_domain
    metrics["num_boundary"] = num_boundary
    metrics["num_initial"] = num_initial
    metrics["n_candidate_points"] = n_candidate_points
    metrics["hard_constrained"] = hard_constrained
    metrics["distribution"] = distribution_based
    metrics["k"] = k
    metrics["c"] = c
    metrics["i"] = 0

    if replace:
        n_sample_points = len(data.train_x_all)

    sampling_f = None
    scoring_f = None

    if method == "random":
        scoring_f = get_random_scores
        sampling_f = partial(sample_random_points, num_points=n_sample_points)
    else:
        if method == "IF":
            scoring_f = get_influence_scores
        elif method == "RAR":
            scoring_f = get_residual_scores
        else:
            raise ValueError(f"Method {method} not recognized")

        if not distribution_based:
            sampling_f = partial(sample_n_from_candidates, num_points=n_sample_points)
        else:
            sampling_f = partial(
                sample_from_distribution,
                num_points=n_sample_points,
                k=k,
                c=c,
            )

    df = pd.DataFrame([metrics])

    for i in range(1):
        scores, candidate_points = scoring_f(
            model,
            data,
            n_candidate_points,
        )
        new_anchors = sampling_f(candidate_points, scores)
        if replace:
            data.replace_with_anchors(new_anchors)
        else:
            data.add_anchors(new_anchors)

        print(f"Iteration {i+1} - len data: {len(data.train_x_all)}")

        model.compile("adam", lr=args.lr)
        model.train(iterations=1000, display_every=100, verbose=1)
        model.compile("L-BFGS")
        model.train(display_every=100, verbose=1)

        metrics = eval_model(model, X_test, y_true)

        metrics["method"] = method
        metrics["seed"] = seed
        metrics["n_iterations"] = n_iterations
        metrics["num_domain"] = num_domain
        metrics["num_boundary"] = num_boundary
        metrics["num_initial"] = num_initial
        metrics["n_candidate_points"] = n_candidate_points
        metrics["hard_constrained"] = hard_constrained
        metrics["distribution"] = distribution_based
        metrics["k"] = k
        metrics["c"] = c
        metrics["i"] = i + 1

        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    loss_history = model.losshistory
    np.savez(
        f"{save_path}/rar_losshistory_{method}_{seed}_hc_{hard_constrained}_distribution_{distribution_based}.npz",
        steps=loss_history.steps,
        loss_train=loss_history.loss_train,
        loss_test=loss_history.loss_test,
    )
    np.savez(
        f"{save_path}/data_{num_domain}_{num_boundary}_{num_initial}_{seed}_iter_{i+i}_{method}.npz",
        train_x_all=data.train_x_all,
        train_x=data.train_x,
        train_x_bc=data.train_x_bc,
        test_x=data.test_x,
    )
    df.to_csv(
        f"{save_path}/rar_metrics_{method}_{seed}_hc_{hard_constrained}_distribution_{distribution_based}.csv",
        index=False,
    )
    model.save(
        f"{save_path}/{os.path.basename(model_name_src.replace('.pt', f'_{method}_{i+1}'))}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
