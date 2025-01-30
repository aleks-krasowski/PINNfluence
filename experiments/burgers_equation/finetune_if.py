import os
import numpy as np
import pandas as pd
import torch
import deepxde as dde
from pathlib import Path
from callbacks import BestModelCheckpoint
from train import burgers_equation, ic, bc, geomtime, gen_testdata
from defaults import DEFAULTS
from _utils.utils import sample_new_training_points_via_IF, get_checkpoint_file


def eval_model(model, x, y, verbose=True):
    y_pred = model.predict(x)
    f = model.predict(x, operator=burgers_equation)
    mean_residual = np.mean(np.abs(f))
    l2_relative_error = dde.metrics.l2_relative_error(y, y_pred)
    mse = dde.metrics.mean_squared_error(y, y_pred)
    if verbose:
        print("Mean residual:", mean_residual)
        print("L2 relative error:", l2_relative_error)
        print("MSE:", mse)
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


def reset_model(data, checkpoint_file):
    net = dde.nn.FNN(DEFAULTS["layers"], "tanh", "Glorot normal")
    chkpt = torch.load(checkpoint_file)
    net.load_state_dict(chkpt["model_state_dict"])
    model = dde.Model(data, net)
    model.compile("adam", lr=DEFAULTS["lr"])

    model.train_state.set_data_train(*model.data.train_next_batch(model.batch_size))
    model.train_state.set_data_test(data.test_x, None, None)
    # model._test()

    epoch = None
    if "epoch" in chkpt.keys():
        epoch = chkpt["epoch"]

    return model, epoch


# Ground truth data
gt_x, gt_y = gen_testdata()

# PDE setup
dde.config.set_random_seed(42)
data = dde.data.TimePDE(
    geomtime,
    burgers_equation,
    ic_bcs=[bc, ic],
    num_domain=DEFAULTS["num_domain"],
    num_boundary=DEFAULTS["num_boundary"],
    num_initial=DEFAULTS["num_initial"],
    num_test=DEFAULTS["num_domain"],
)

# Load initial checkpoint
checkpoint_file = get_checkpoint_file(DEFAULTS["save_path"], "adam_50000_best")
checkpoint_filename = Path(checkpoint_file).stem
if not os.path.exists(f"{DEFAULTS['save_path']}/{checkpoint_filename}"):
    os.mkdir(f"{DEFAULTS['save_path']}/{checkpoint_filename}")

net = dde.nn.FNN(DEFAULTS["layers"], "tanh", "Glorot normal")
net.load_state_dict(torch.load(checkpoint_file)["model_state_dict"])

reset_model(data, checkpoint_file)

# Load data if available
data_file = "./model_zoo/data_points.npz"
if os.path.exists(data_file):
    data_points = np.load(data_file)
    reset_data(data, data_points)

# Load influence file or generate
if_file = f"{DEFAULTS['save_path']}/{checkpoint_filename}/influences.npz"
if os.path.exists(if_file):
    influences_arrs = np.load(if_file)
    influences = influences_arrs["influences"]
    new_train_points = influences_arrs["new_train_points"]
    OG_train_points = influences_arrs["OG_train_points"]
else:
    influences, new_train_points, OG_train_points = sample_new_training_points_via_IF(
        net,
        data,
        show_progress=False,
        fraction_of_train=5,
    )
    np.savez_compressed(
        f"{DEFAULTS['save_path']}/{checkpoint_filename}/influences",
        influences=influences,
        new_train_points=new_train_points,
        OG_train_points=OG_train_points,
    )

summed_abs_influences = np.abs(influences).sum(axis=0)
top_idx = np.argsort(summed_abs_influences)[::-1]

setting = "add_influential_data"

if setting == "sample_new_points":
    chosen_pants = new_train_points[top_idx[: len(OG_train_points)]]
    for i, boundary_cond in enumerate(data.bcs):
        if isinstance(boundary_cond, dde.IC):
            on_boundary = boundary_cond.on_initial(
                chosen_pants, data.geom.on_initial(chosen_pants)
            )
            if not on_boundary.any():
                boundary_point = boundary_cond.geom.random_initial_points(1)
                chosen_pants = np.concatenate([chosen_pants, boundary_point])
        else:
            on_boundary = boundary_cond.on_boundary(
                chosen_pants, data.geom.on_boundary(chosen_pants)
            )
            if not on_boundary.any():
                boundary_point = boundary_cond.geom.random_boundary_points(1)
                chosen_pants = np.concatenate([chosen_pants, boundary_point])

    data.train_x_all = chosen_pants
    data.train_x = data.train_y = data.train_aux_vars = data.train_x_bc = None
    data.train_next_batch()

    model = reset_model(data, checkpoint_file)
    model.compile("adam", lr=DEFAULTS["lr"] * 0.1)
    model.train(iterations=50_000, display_every=1000)
    model.save(f"{DEFAULTS['save_path']}/finetuned_influence")

elif setting == "add_influential_data":
    ratios = [0] + list(np.arange(0.01, 0.11, 0.01)) + list(np.arange(0.1, 1.1, 0.1))
    # ratios = [0.5]
    # # ratios = [0]
    metrics_df = pd.DataFrame(
        columns=["model", "mean_residual", "l2_relative_error", "mse", "ratio"]
    )

    for ratio in ratios:
        print(f"ratio: {ratio}")
        print("IF")
        data = reset_data(data, data_points)
        model, epoch = reset_model(data, checkpoint_file)
        # model._test()

        data.add_anchors(new_train_points[top_idx[: int(len(OG_train_points) * ratio)]])
        # TODO: respect boundary and initial conditions without changing behaviour on boundary
        # model._test()
        # data.train_x = data.train_y = data.train_aux_vars = data.train_x_bc = None
        # data.train_next_batch()
        # model._test()
        model.train(
            iterations=10_000,
            display_every=1_000,
            callbacks=[
                BestModelCheckpoint(
                    f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_if_{setting}_{ratio}_test.pt",
                    verbose=1,
                    save_better_only=True,
                ),
                BestModelCheckpoint(
                    f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_if_{setting}_{ratio}_train.pt",
                    verbose=1,
                    save_better_only=True,
                    monitor="train loss",
                ),
            ],
        )
        model.save(
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_if_{setting}_{ratio}_full_epochs"
        )

        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "if"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "full_epochs"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        model, epoch = reset_model(
            data,
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_if_{setting}_{ratio}_test.pt",
        )
        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "if"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "best_test_loss"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        model, epoch = reset_model(
            data,
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_if_{setting}_{ratio}_train.pt",
        )
        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "if"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "best_train_loss"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        metrics_df.to_csv(f"{DEFAULTS['save_path']}/{checkpoint_filename}/metrics.csv")

        print("Random")
        dde.config.set_random_seed(42)
        new_train_points_random = data.geom.random_points(
            int(len(OG_train_points) * ratio)
        )
        data = reset_data(data, data_points)
        data.add_anchors(new_train_points_random)

        model, epoch = reset_model(data, checkpoint_file)
        model.train(
            iterations=10_000,
            display_every=1_000,
            callbacks=[
                BestModelCheckpoint(
                    f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_random_{setting}_{ratio}_test.pt",
                    verbose=1,
                    save_better_only=True,
                ),
                BestModelCheckpoint(
                    f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_random_{setting}_{ratio}_train.pt",
                    verbose=1,
                    save_better_only=True,
                    monitor="train loss",
                ),
            ],
        )

        model.save(
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_random_{setting}_{ratio}_full_epochs"
        )

        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "random"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "full_epochs"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        model, epoch = reset_model(
            data,
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_random_{setting}_{ratio}_test.pt",
        )
        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "random"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "best_test_loss"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        model, epoch = reset_model(
            data,
            f"{DEFAULTS['save_path']}/{checkpoint_filename}/finetuned_random_{setting}_{ratio}_train.pt",
        )
        metrics = eval_model(model, gt_x, gt_y)
        metrics["model"] = "random"
        metrics["ratio"] = ratio
        metrics["epoch"] = epoch
        metrics["model_version"] = "best_train_loss"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
