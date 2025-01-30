import deepxde as dde
import numpy as np
from deepxde.backend import tf
import torch

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../..")
from _utils.utils import sample_new_training_points_via_IF, get_checkpoint_file

# dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)
dde.config.set_random_seed(42)


def gen_testdata():
    data = np.load("./dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def main():
    NumDomain = 2000
    save_dir = "./model_zoo/like_rar/"
    save_dir_finetuned = os.path.join(save_dir, "finetuned_50")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if not os.path.isdir(save_dir_finetuned):
        os.makedirs(save_dir_finetuned)

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    X_test, y_true = gen_testdata()
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(
        geomtime, pde, [], num_domain=NumDomain // 2, train_distribution="pseudo"
    )

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -torch.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)

    if not os.path.exists(f"{save_dir}/adam-15000.pt"):
        model.train(epochs=15000)
        model.save(f"{save_dir}/adam")
    else:
        model.restore(f"{save_dir}/adam-15000.pt")
    # model.compile("L-BFGS")
    # if not os.path.exists(f"{save_dir}/lbfgs-16000.pt"):
    #     model.train()
    #     model.save(f"{save_dir}/lbfgs")
    # else:
    #     model.restore(f"{save_dir}/lbfgs-16000.pt")

    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error = [np.array([l2_error])]
    print(f"l2_relative_error: {l2_error:.2e}")

    for i in range(100):
        print(i)

        chkpt = get_checkpoint_file(save_dir_finetuned, f"{i:04d}")
        if chkpt is None:
            # X = geomtime.random_points(100000)
            # Y = np.abs(model.predict(X, operator=pde))[:, 0]
            influences, new_train_points, OG_train_points = (
                sample_new_training_points_via_IF(
                    model.net, data, show_progress=False, num_domain=10_000
                )
            )
            summed_abs_influences = np.abs(influences).sum(axis=0)
            top_idx = np.argsort(summed_abs_influences)[::-1]
            # err_eq = torch.tensor(Y)
            data.add_anchors(new_train_points[top_idx[: (NumDomain // 50)]])

            model.compile("adam", lr=0.001)
            model.train(epochs=1000)
            # model.compile("L-BFGS")
            # losshistory, train_state = model.train()
            model.save(os.path.join(save_dir_finetuned, f"{i:04d}"))

        else:
            model.restore(chkpt)

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error.append(np.array([l2_error]))
        print(f"l2_relative_error: {l2_error:.2e}")

        error_np = np.array(error)
        np.savetxt(f"error_if.txt", error)

    error = np.array(error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f"error_RAR-G.txt", error)
    return error


if __name__ == "__main__":
    main()
