import argparse
import captum
import deepxde as dde
import numpy as np
import torch
from pathlib import Path
from train import (
    navier_stokes,
    navier_stokes_broken,
    geom,
    bc_inflow_u,
    bc_inflow_v,
    bc_noslip_u,
    bc_noslip_v,
    bc_outflow_u,
    bc_outflow_v,
    L,
    W,
    cylinder_center,
    radius
)

import sys 
sys.path.append("../..")

from _utils.models import PINNLoss, NetPredWrapper, NetworkWithPDE, ScaledFNN
from _utils.dataset import DummyDataset
from _utils.utils import set_default_device

set_default_device("mps")

dde.config.set_random_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Influence Function')
    parser.add_argument("--checkpoint", type=str, default="./model_zoo/good/lbfgs-125000.pt", help="Model checkpoint to use")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 64, 64, 64, 64, 3], help="Layer sizes")
    parser.add_argument("--batch_size", type=int, default=None, help='batch size for the dataloaders')
    parser.add_argument("--train_x_path", type=str, help='Path to numpy file containing training data', required=True)
    parser.add_argument("--broken", action="store_true", help="Use broken PDE")
    parser.add_argument('--save_path', type=str, default='./model_zoo/influences', help='Path to save model')
    return parser.parse_args()

def main(args):
    print(args)

    chkpt_path = args.checkpoint
    layers = args.layers
    batch_size = args.batch_size
    train_x_path = args.train_x_path
    broken = args.broken
    save_path = Path(args.save_path)

    device = torch.get_default_device()

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_load_func = lambda model, path: model.load_state_dict(
        torch.load(path, map_location=device)['model_state_dict'])

    checkpoint_load_func_pde_net = lambda model, path: model.pinn.load_state_dict(
        torch.load(path, map_location=device)['model_state_dict'])


    net = ScaledFNN(layers, "gelu", "Glorot uniform", L, W)
    checkpoint_load_func(net, chkpt_path)
    net.eval()

    continuity = lambda xy, out: navier_stokes(xy, out)[0]
    momentum_x = lambda xy, out: navier_stokes(xy, out)[1]
    momentum_y = lambda xy, out: navier_stokes(xy, out)[2]
    inflow_boundary = lambda x: torch.isclose(x[:, 0], torch.tensor(0, dtype=torch.float32))
    outflow_boundary = lambda x: torch.isclose(x[:, 0], torch.tensor(L, dtype=torch.float32))
    noslip_boundary = lambda x: torch.logical_or(
        torch.logical_or(
            torch.isclose(x[:, 1], torch.tensor(0, dtype=torch.float32)),
            torch.isclose(x[:, 1], torch.tensor(W, dtype=torch.float32))
        ),
        (torch.sqrt((x[:, 0] - cylinder_center[0]) ** 2 + (x[:, 1] - cylinder_center[1]) ** 2) <= radius)
    )

    pde_net = NetworkWithPDE(
        pinn=net,
        pdes=[
            continuity,
            momentum_x if not broken else lambda xy, out: navier_stokes_broken(xy, out)[1],
            momentum_y,
        ],
        bcs=[
            (bc_inflow_u, inflow_boundary),
            (bc_inflow_v, inflow_boundary),
            (bc_outflow_u, outflow_boundary),
            (bc_outflow_v, outflow_boundary),
            (bc_noslip_u, noslip_boundary),
            (bc_noslip_v, noslip_boundary),
        ],
        geom=geom
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
        checkpoints_load_func=lambda x,y: 0, # already loaded thus dummy
        batch_size=batch_size
    )


    arrs = {}

    test_x = np.array(np.load("./dataset/ns_steady.npy", allow_pickle=True).item()['coords']).astype(np.float32)
    print(f"test data shape: {test_x.shape}")
    testset = DummyDataset(test_x, return_zeroes=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=max(batch_size, len(testset)), shuffle=False)

    # calculate f(x) influence for model output dimensions
    # fourth dimension is vector norm of 0 and 1 (|||vec{u}|| = sqrt{u_1^2 + u_2^2})
    if_instance.model_test = NetPredWrapper(net, 0)
    if_instance.test_loss_fn = None

    arrs["train_x"] = train_x
    arrs["test_x"] = test_x

    for output_dim in range(0,4):
        if_instance.model_test = NetPredWrapper(net, output_dim)
        print(f"output dim: {output_dim}")
        arrs[f"output_dim_{output_dim}"] = if_instance.influence(testloader, show_progress=True)

    # calculate loss influences
    if_instance.model_test = pde_net

    print("pde")
    if_instance.test_loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[0,1,2])
    if_instance.test_reduction_type = "none"
    influences_pde = if_instance.influence(testloader, show_progress=True)
    arrs["influences_pde"] = influences_pde

    print("bcs")
    if_instance.test_loss_fn = PINNLoss(include_all_losses=False, include_specific_ids=[3, 4, 5, 6, 7, 8])
    if_instance.test_reduction_type = "none"
    influences_bcs = if_instance.influence(testloader, show_progress=True)
    arrs["bcs"] = influences_bcs

    print("all_loss_terms")
    if_instance.test_loss_fn = PINNLoss()
    if_instance.test_reduction_type = "none"
    influences_bcs = if_instance.influence(testloader, show_progress=True)
    arrs["all_loss_terms"] = influences_bcs

    np.savez_compressed(str(save_path / "influence_arrs"), **arrs)

if __name__ == "__main__":
    args = parse_args()
    main(args)