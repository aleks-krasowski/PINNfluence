import argparse
from pathlib import Path

import deepxde as dde
import numpy as np

import sys

sys.path.append("../..")

from _utils.models import ScaledFNN
from _utils.utils import StopOnBrokenLBFGS, set_default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Burgers Equations Solver")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--layers', type=int, nargs='+', default=[2] + [20] * 3 + [1], help='Layer sizes')
    parser.add_argument('--n_iterations', type=int, default=50_000, help='Number of iterations')
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
    geomtime, lambda x: -dde.backend.torch.sin(dde.backend.torch.pi * x[:, 0:1]), lambda _, on_initial: on_initial
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

    net = dde.nn.FNN(args.layers, "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    model.train(
        iterations=n_iter,
        display_every=1000,
    )

    model.save(f"{save_path}/adam")

    stop_on_broken = StopOnBrokenLBFGS()
    dde.optimizers.config.set_LBFGS_options(
        maxiter=n_iter_lbfgs,
    )
    model.compile("L-BFGS")
    model.train(display_every=1, callbacks=[stop_on_broken])

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
