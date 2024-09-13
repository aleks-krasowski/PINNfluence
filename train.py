import argparse
from pathlib import Path

import deepxde as dde
import numpy as np

from _utils.models import ScaledFNN


def parse_args():
    parser = argparse.ArgumentParser(description="Navier-Stokes Solver")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 64, 64, 64, 64, 3], help='Layer sizes')
    parser.add_argument('--n_iterations', type=int, default=100_000, help='Number of iterations')
    parser.add_argument('--n_iterations_lbfgs', type=int, default=25_000, help='Number of iterations')
    parser.add_argument('--num_domain', type=int, default=7_500, help='Number of domain points')
    parser.add_argument('--num_boundary', type=int, default=2_500, help='Number of boundary points')
    parser.add_argument('--save_path', type=str, default='./model_zoo', help='Path to save model')
    parser.add_argument("--broken", action="store_true", help="Use broken navier stokes equation")
    return parser.parse_args()


class StopOnBrokenLBFGS(dde.callbacks.Callback):
    """
    This callback implements a mechanism to stop L-BFGS optimization upon it breaking.
    This may be useful due, to an unfortunately long known bug in PyTorchs implementation
    of L-BFGS.
    In some cases NaN values are produced during the approximation of the Hessian.
    See:
        - https://github.com/lululxvi/deepxde/issues/1605
        - https://github.com/pytorch/pytorch/issues/5953
    """

    def __init__(self):
        super().__init__()
        self.prev_n_iter = 0

    def on_epoch_end(self):
        n_iter = self.model.opt.state_dict()["state"][0]["n_iter"]

        # if only one iteration was executed although more are defined we assume that something broke
        if (n_iter - self.prev_n_iter == 1) and (dde.optimizers.LBFGS_options["iter_per_step"] != 1):
            self.model.stop_training = True
            print("Encountered broken LBFGS. Stopping training.")

        self.prev_n_iter = n_iter


dde.config.set_random_seed(42)


def navier_stokes(xy, out):
    """Navier-Stokes equation"""
    rho = 1.0

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    du_x = dde.grad.jacobian(out, xy, i=0, j=0)
    dv_x = dde.grad.jacobian(out, xy, i=1, j=0)
    dp_x = dde.grad.jacobian(out, xy, i=2, j=0)

    du_y = dde.grad.jacobian(out, xy, i=0, j=1)
    dv_y = dde.grad.jacobian(out, xy, i=1, j=1)
    dp_y = dde.grad.jacobian(out, xy, i=2, j=1)

    du_xx = dde.grad.hessian(out, xy, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(out, xy, component=0, i=1, j=1)

    dv_xx = dde.grad.hessian(out, xy, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(out, xy, component=1, i=1, j=1)

    continuity = du_x + dv_y
    x_momentum = u * du_x + v * du_y + 1 / rho * dp_x - nu * (du_xx + du_yy)
    y_momentum = u * dv_x + v * dv_y + 1 / rho * dp_y - nu * (dv_xx + dv_yy)

    return [continuity, x_momentum, y_momentum]


def navier_stokes_broken(xy, out):
    """Navier-Stokes equation"""
    rho = 1.0
    nu = 0.001
    eps = 1e-8

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    du_x = dde.grad.jacobian(out, xy, i=0, j=0)
    dv_x = dde.grad.jacobian(out, xy, i=1, j=0)
    dp_x = dde.grad.jacobian(out, xy, i=2, j=0)

    du_y = dde.grad.jacobian(out, xy, i=0, j=1)
    dv_y = dde.grad.jacobian(out, xy, i=1, j=1)
    dp_y = dde.grad.jacobian(out, xy, i=2, j=1)

    du_xx = dde.grad.hessian(out, xy, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(out, xy, component=0, i=1, j=1)

    dv_xx = dde.grad.hessian(out, xy, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(out, xy, component=1, i=1, j=1)

    continuity = du_x + dv_y
    x_momentum = v * du_y + 1 / rho * dp_x - nu * (du_xx + du_yy)
    y_momentum = u * dv_x + v * dv_y + 1 / rho * dp_y - nu * (dv_xx + dv_yy)

    return [continuity, x_momentum, y_momentum]


# define geometry (domain)
L = 2.2
W = 0.41
U_max = 0.3
nu = 1e-3
geom_space = dde.geometry.Rectangle([0, 0], [L, W])
cylinder_center = np.array([0.2, 0.2])
radius = 0.05
cylinder = dde.geometry.Disk(cylinder_center, radius)
geom = geom_space - cylinder

# define boundary conditions
bc_noslip_u = dde.DirichletBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
            np.isclose(x[1], 0)
            or np.isclose(x[1], W)
            or (np.sqrt((x[0] - cylinder_center[0]) ** 2 + (x[1] - cylinder_center[1]) ** 2) <= radius)
    ),
    component=0
)
bc_noslip_v = dde.DirichletBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and (
            np.isclose(x[1], 0)
            or np.isclose(x[1], W)
            or (np.sqrt((x[0] - cylinder_center[0]) ** 2 + (x[1] - cylinder_center[1]) ** 2) <= radius)
    ),
    component=1
)


def parabolic_inflow_u(xy):
    y = xy[:, 1:2]
    u = 4 * U_max * y * (W - y) / (W ** 2)
    return u


def outflow_u(X, y, _):
    u_x = dde.grad.jacobian(y, X, i=0, j=0)
    p = y[:, 2:3]
    return nu * u_x - p


def outflow_v(X, y, _):
    u_y = dde.grad.jacobian(y, X, i=0, j=1)
    return nu * u_y


bc_inflow_u = dde.DirichletBC(
    geom,
    parabolic_inflow_u,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    component=0
)
bc_inflow_v = dde.DirichletBC(
    geom,
    lambda x: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    component=1
)
bc_outflow_u = dde.OperatorBC(
    geom,
    outflow_u,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], L)
)
bc_outflow_v = dde.OperatorBC(
    geom,
    outflow_v,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], L)
)

bcs = [bc_inflow_u, bc_inflow_v, bc_outflow_u, bc_outflow_v, bc_noslip_u, bc_noslip_v]


def main(args):
    print(args)
    lr = args.lr
    layers = args.layers
    num_domain = args.num_domain
    num_boundary = args.num_boundary
    save_path = Path(args.save_path)
    n_iter = args.n_iterations
    n_iter_lbfgs = args.n_iterations_lbfgs
    broken = args.broken

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if broken:
        pde = navier_stokes_broken
    else:
        pde = navier_stokes

    data = dde.data.PDE(
        geom,
        pde,
        bcs=bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test=2_500
    )

    net = ScaledFNN(layers, "gelu", "Glorot uniform", L, W)

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


if __name__ == "__main__":
    args = parse_args()
    main(args)
