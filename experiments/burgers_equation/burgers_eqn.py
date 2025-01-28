import argparse
from gc import callbacks
from pathlib import Path

import deepxde as dde
import numpy as np

import sys

from torch.backends.mkl import verbose

sys.path.append("../..")

from _utils.models import ScaledFNN
from _utils.utils import StopOnBrokenLBFGS, set_default_device

from deepxde.callbacks import Callback
import torch


class BestModelCheckpoint(Callback):
    """Save the PyTorch model after every epoch, with the annoying appending of DeepXDE.

    Args:
        filepath (str): Path to save the model file.
        verbose (int): Verbosity mode, 0 or 1.
        save_better_only (bool): If True, only saves the model if the monitored
            metric improves.
        monitor (str): The metric to monitor ('train_loss' or 'val_loss').
        min_delta (float): Minimum change to qualify as an improvement.
    """

    def __init__(
            self, filepath, verbose=0, save_better_only=False, monitor="train loss", min_delta=0.0
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = np.inf

    def on_epoch_end(self):
        """Save the model at the end of an epoch."""
        current = self.get_monitor_value()
        epoch = self.model.train_state.epoch

        if current is None:
            raise ValueError(f"Metric '{self.monitor}' is not available in logs.")

        if self.save_better_only:
            if current < self.best - self.min_delta:
                self.best = current
                self._save_model(epoch, current)
        else:
            self._save_model(epoch, current)

    def _save_model(self, epoch, current):
        """Save the model to the specified filepath."""
        if self.verbose > 0:
            print(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}, saving model to {self.filepath}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.net.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
        }
        torch.save(checkpoint, self.filepath)

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Burgers Equations Solver")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--layers', type=int, nargs='+', default=[2] + [20] * 3 + [1], help='Layer sizes')
    parser.add_argument('--n_iterations', type=int, default=100_000, help='Number of iterations')
    parser.add_argument('--n_iterations_lbfgs', type=int, default=12_500, help='Number of iterations')
    parser.add_argument('--num_domain', type=int, default=2540, help='Number of domain points')
    parser.add_argument('--num_boundary', type=int, default=80, help='Number of boundary points')
    parser.add_argument('--num_initial', type=int, default=160, help='Number of initial points')
    parser.add_argument('--save_path', type=str, default='./model_zoo', help='Path to save model')
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    return parser.parse_args()


dde.config.set_random_seed(42)


def gen_testdata():
    data = np.load("dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


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
    geomtime, lambda x: -np.sin(dde.backend.torch.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)


def main(args):
    print(args)
    lr = args.lr
    layers = args.layers
    num_domain = args.num_domain
    num_boundary = args.num_boundary
    num_initial = args.num_initial
    save_path = Path(args.save_path)
    n_iter = args.n_iterations
    n_iter_lbfgs = args.n_iterations_lbfgs
    device = args.device

    set_default_device(device)

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    pde = burgers_equation

    data = dde.data.TimePDE(
        geomtime,
        pde,
        ic_bcs=[bc, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=int(1 / 4 * num_domain),
    )

    net = dde.nn.FNN(layers, "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(
        iterations=n_iter,
        display_every=1000,
        callbacks=[BestModelCheckpoint(f"{save_path}/adam_best.pt", verbose=1, save_better_only=True)],
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    model.save(f"{save_path}/adam")

    dde.optimizers.config.set_LBFGS_options(
        maxiter=n_iter_lbfgs,
    )
    model.compile("L-BFGS")
    losshistory, train_state = model.train(display_every=1,
                                           callbacks=[BestModelCheckpoint(f"{save_path}/lbfgs_best.pt", verbose=1,
                                                                          save_better_only=True)])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    model.save(f"{save_path}/lbfgs")
    np.save(f"{save_path}/train_x", data.train_x_all)

    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    f = model.predict(X, operator=burgers_equation)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))


if __name__ == "__main__":
    args = parse_args()
    main(args)
