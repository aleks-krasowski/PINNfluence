import deepxde as dde
import numpy as np
import torch
from train import burgers_equation, ic, bc, geomtime
from defaults import DEFAULTS

from _utils.utils import sample_new_training_points_via_IF, get_checkpoint_file

net = dde.nn.FNN(DEFAULTS["layers"], "tanh", "Glorot normal")

pde = burgers_equation
dde.config.set_random_seed(42)
data = dde.data.TimePDE(
    geomtime,
    pde,
    ic_bcs=[bc, ic],
    num_domain=DEFAULTS["num_domain"],
    num_boundary=DEFAULTS["num_boundary"],
    num_initial=DEFAULTS["num_initial"],
    num_test=DEFAULTS["num_domain"],
)

checkpoint_file = get_checkpoint_file(DEFAULTS["save_path"], "lbfgs")
net.load_state_dict(torch.load(checkpoint_file)["model_state_dict"])

influences, new_train_points, OG_train_points = sample_new_training_points_via_IF(
    net, data, show_progress=False
)

summed_abs_influences = np.abs(influences.numpy()).sum(axis=0)
top_idx = np.argsort(summed_abs_influences)[::-1]

data.train_x_all = new_train_points[top_idx[len(data.train_x_all) :]]
data.bc_points()

model = dde.Model(data, net)
model.compile("adam", lr=DEFAULTS["lr"])
model.train(iterations=10_000, display_every=1000)
model.save(f"{DEFAULTS['save_path']}/finetuned_influence")
