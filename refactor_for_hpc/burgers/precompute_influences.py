import os
import numpy as np
import argparse
import deepxde as dde
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import sample_new_training_points_via_IF
from pretrain_model import (
    burgers_equation,
    geomtime,
    bc,
    ic,
    parse_args,
)
from eval_finetuning import reset_data
from defaults import DEFAULTS


def create_pde_data(args):
    # replicate data creation from the pretrain script
    data = dde.data.TimePDE(
        geomtime,
        burgers_equation,
        ic_bcs=[bc, ic] if not args.hard_constrained else [],
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        num_test=10_000,
    )
    return data


def load_trained_model(args, data):
    # same network definition
    net = dde.maps.FNN(args.layers, "tanh", "Glorot normal")

    # apply output transform if hard_constrained
    if args.hard_constrained:

        def output_transform(x, y):
            return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

        net.apply_output_transform(output_transform)

        ckpt_name = f"/opt/model_zoo_src/adam_{args.n_iterations}_best_seed_{args.seed}_hardconstrained.pt"
    else:
        ckpt_name = (
            f"/opt/model_zoo_src/adam_{args.n_iterations}_best_seed_{args.seed}.pt"
        )

    net.load_state_dict(torch.load(ckpt_name)["model_state_dict"])
    return net


if __name__ == "__main__":
    # all argument combinations from the array list
    args = parse_args()
    lr = args.lr
    layers = args.layers
    n_iterations = args.n_iterations
    num_domain = args.num_domain
    num_boundary = args.num_boundary
    num_initial = args.num_initial
    save_path = args.save_path
    seed = args.seed
    hard_constrained = args.hard_constrained

    # create data and model
    data = create_pde_data(args)
    data_points = np.load(
        f"/opt/model_zoo_src/data_{num_domain}_{num_boundary}_{num_initial}_{seed}.npz"
    )
    reset_data(data, data_points)

    net = load_trained_model(args, data)
    print("Model loaded")
    # compute influence scores and candidate points
    infl_scores, candidate_points, orig_train_points = (
        sample_new_training_points_via_IF(
            net=net,
            data=data,
            fraction_of_train=10_000 / len(data.train_x_all),
            seed=seed,
        )
    )

    np.savez_compressed(
        f"{save_path}/influence_scores_{n_iterations}_{num_domain}_{num_boundary}_{num_initial}_{seed}_hc_{hard_constrained}.npz",
        influence_scores=infl_scores,
        candidate_points=candidate_points,
        original_train_points=orig_train_points,
    )
