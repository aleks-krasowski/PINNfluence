import argparse
import captum
import deepxde as dde
import numpy as np
import torch
from pathlib import Path
from burgers_eqn import burgers_equation, ic, bc

import sys

sys.path.append("../..")

from _utils.models import PINNLoss, NetPredWrapper, ModelWrapper, ScaledFNN
from _utils.dataset import DummyDataset
from _utils.utils import set_default_device

set_default_device("cpu")

dde.config.set_random_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Influence Function")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./model_zoo/lbfgs-55079.pt",
        help="Model checkpoint to use",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2] + [20] * 3 + [1],
        help="Layer sizes",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size for the dataloaders"
    )
    parser.add_argument(
        "--train_x_path",
        type=str,
        default="./model_zoo/train_x.npy",
        help="Path to numpy file containing training data",
    )
    parser.add_argument("--broken", action="store_true", help="Use broken PDE")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model_zoo/influences",
        help="Path to save model",
    )
    return parser.parse_args()


def main(args):
    print(args)

    chkpt_path = args.checkpoint
    layers = args.layers
    batch_size = args.batch_size
    train_x_path = args.train_x_path
    save_path = Path(args.save_path)

    device = torch.get_default_device()

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_load_func = lambda model, path: model.load_state_dict(
        torch.load(path, map_location=device)["model_state_dict"]
    )

    net = dde.nn.FNN(layers, "tanh", "Glorot normal")
    checkpoint_load_func(net, chkpt_path)
    net.eval()

    pde_net = ModelWrapper(
        net=net, pde=burgers_equation, bcs=[ic, bc]
    )

    train_x = np.load(train_x_path)
    print(f"train data shape: {train_x.shape}")
    trainset = DummyDataset(train_x, return_zeroes=True)

    if batch_size is None:
        batch_size = len(trainset)

    if_instance = captum.influence.ArnoldiInfluenceFunction(
        pde_net,
        train_dataset=trainset,
        checkpoint=args.checkpoint,
        loss_fn=PINNLoss(),
        show_progress=True,
        projection_on_cpu=False,
        checkpoints_load_func=lambda x, y: 0,  # already loaded thus dummy
        batch_size=batch_size,
    )

    arrs = {}
    data = np.load("dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    test_x = np.vstack((np.ravel(xx), np.ravel(tt)),dtype=np.float32).T
    # random subset of test_x
    test_x = test_x[np.random.choice(test_x.shape[0], 1000, replace=False)]
    print(f"test data shape: {test_x.shape}")
    testset = DummyDataset(test_x, return_zeroes=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=min(batch_size, len(testset)), shuffle=False
    )

    # calculate f(x) influence for model output dimensions
    # fourth dimension is vector norm of 0 and 1 (|||vec{u}|| = sqrt{u_1^2 + u_2^2})
    if_instance.model_test = NetPredWrapper(net, 0)
    if_instance.test_loss_fn = None

    arrs["train_x"] = train_x
    arrs["test_x"] = test_x

    for output_dim in range(0, 1):
        if_instance.model_test = NetPredWrapper(net, output_dim)
        print(f"output dim: {output_dim}")
        arrs[f"output_dim_{output_dim}"] = if_instance.influence(
            testloader, show_progress=True
        )

    # calculate loss influences
    if_instance.model_test = pde_net

    print("pde")
    if_instance.test_loss_fn = PINNLoss(
        include_all_losses=False, include_specific_ids=[0]
    )
    if_instance.test_reduction_type = "none"
    influences_pde = if_instance.influence(testloader, show_progress=True)
    arrs["influences_pde"] = influences_pde

    print("ic")
    if_instance.test_loss_fn = PINNLoss(
        include_all_losses=False, include_specific_ids=[1]
    )
    if_instance.test_reduction_type = "none"
    influences_bcs = if_instance.influence(testloader, show_progress=True)
    arrs["ic"] = influences_bcs

    print("bc")
    if_instance.test_loss_fn = PINNLoss(
        include_all_losses=False, include_specific_ids=[2]
    )
    if_instance.test_reduction_type = "none"
    influences_bcs = if_instance.influence(testloader, show_progress=True)
    arrs["bc"] = influences_bcs

    print("all_loss_terms")
    if_instance.test_loss_fn = PINNLoss()
    if_instance.test_reduction_type = "none"
    influences_bcs = if_instance.influence(testloader, show_progress=True)
    arrs["all_loss_terms"] = influences_bcs

    np.savez_compressed(str(save_path / "influence_arrs_new"), **arrs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
