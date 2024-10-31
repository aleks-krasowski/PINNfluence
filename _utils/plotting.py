import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import matplotlib.patches as patches


dim_names = {
    0: "Predicted $u_1$",
    1: "Predicted $u_2$",
    2: "Predicted $p$",
    3: "Predicted $||\\vec{u}||$",
    "influences_pde": "pde_loss",
    "bcs": "bc_loss",
    "all_loss_terms": "all_losses"
}

L = 2.2
W = 0.41
cylinder_center = np.array([0.2, 0.2])
radius = 0.05

base_fig_size = (L * 10, W * 10)

def set_axes_limit(ax):
    ax.set_xlim(-L / 20, L + L / 20)
    ax.set_ylim(-W / 20, W + W / 20)

def scatter_with_colorbar(
        fig,
        ax,
        xy,
        c,
        cmap='jet',
        markersize=8,
        norm=None,
        limit_axes = True
):
    if norm:
        absmax = np.abs(c).max()
        norm = TwoSlopeNorm(vmin=-absmax, vmax=absmax, vcenter=0)
    sc = ax.scatter(*xy.T, c=c, cmap=cmap, s=markersize, norm=norm)
    fig.colorbar(sc, ax=ax)
    if limit_axes:
        set_axes_limit(ax)

def plot_results(net, coords, u_target, v_target, p_target, cmap='jet', markersize=8):
    # Create a grid of points within the bounding box of the polygon
    # Predict the results
    x = torch.as_tensor(coords, dtype=torch.float32)
    u_pred, v_pred, p_pred = net(x).detach().T

    error_u = np.abs(u_pred - u_target)
    error_v = np.abs(v_pred - v_target)
    error_uv = np.abs((u_pred**2 + v_pred**2) - (u_target**2 + v_target**2))
    error_p = np.abs(p_pred - p_target)

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(base_fig_size[0] * 3, base_fig_size[1] * 4))

    # Plot the predicted u velocity
    scatter_with_colorbar(fig, ax[0,0], coords, u_pred, cmap, markersize)
    ax[0, 0].set_title('Predicted $u_1$')

    scatter_with_colorbar(fig, ax[0,1], coords, u_target, cmap, markersize)
    ax[0, 1].set_title('Target $u_1$')

    scatter_with_colorbar(fig, ax[0, 2], coords, error_u, cmap, markersize)
    ax[0, 2].set_title('Absolute error in $u_1$')

    # Plot the predicted v velocity
    scatter_with_colorbar(fig, ax[1, 0], coords, v_pred, cmap, markersize)
    ax[1, 0].set_title('Predicted $u_2$')

    scatter_with_colorbar(fig, ax[1, 1], coords, v_target, cmap, markersize)
    ax[1, 1].set_title('Target $u_2$')

    scatter_with_colorbar(fig, ax[1, 2], coords, error_v, cmap, markersize)
    ax[1, 2].set_title('Absolute error in $u_2$')

    scatter_with_colorbar(fig, ax[2, 0], coords, np.sqrt(u_pred ** 2 + v_pred ** 2), cmap, markersize)
    ax[2, 0].set_title('Predicted $||\\vec{u}||$')

    scatter_with_colorbar(fig, ax[2, 1], coords, np.sqrt(u_target ** 2 + v_target ** 2), cmap, markersize)
    ax[2, 1].set_title('Target $||\\vec{u}||$')

    scatter_with_colorbar(fig, ax[2, 2], coords, error_uv, cmap, markersize)
    ax[2, 2].set_title('Absolute error in $||\\vec{u}||$')

    # Plot the predicted pressure
    scatter_with_colorbar(fig, ax[3, 0], coords, p_pred, cmap, markersize)
    ax[3, 0].set_title('Predicted $p$')

    scatter_with_colorbar(fig, ax[3, 1], coords, p_target, cmap, markersize)
    ax[3, 1].set_title('Target $p$')

    scatter_with_colorbar(fig, ax[3, 2], coords, error_p, cmap, markersize)
    ax[3, 2].set_title('Absolute error in $p$')

    fig.tight_layout()
    return fig, ax

def plot_losses(
        pde_net: torch.nn.Module,
        loss_fn: torch.nn.Module,
        x: torch.Tensor
):
    assert loss_fn.reduction == "none", "loss_fn.reduction must be set to 'none'"
    losses = loss_fn(pde_net(x), None).sum(axis=1).detach()
    fig, ax = plt.subplots(figsize = base_fig_size)
    scatter_with_colorbar(fig, ax, x.detach(), np.log(losses))
    return fig, ax

def plot_influences_heatmap_for_train_point(
        fig,
        ax,
        coords,
        preds, # expected to be shape of [n, 4] with 4 being total number of dimensions
        influence_arrs,
        indices,
        index_key,
        model_key,
        dim = 0,
        markersize=8
):
    if not isinstance(dim, int):
        dim_key = dim
        preds_to_plot = preds[model_key][3]
    else:
        dim_key = f"output_dim_{dim}"
        preds_to_plot = preds[model_key][dim]
    idx = indices[model_key][index_key]
    scatter_with_colorbar(fig, ax[0], coords, preds_to_plot)
    scatter_with_colorbar(fig, ax[1], coords, influence_arrs[model_key][dim_key][:, idx], norm=True,
                          cmap="bwr")
    ax[1].scatter(*influence_arrs[model_key]['train_x'][idx].T, marker='x', c='k', s=200)

def plot_influence_heatmap_for_area(
        fig,
        ax,
        coords,
        preds,
        influence_arrs,
        aoi_name,
        aoi_selector,
        model_key,
        dim = 0,
        markersize = 8,
):

    if not isinstance(dim, int):
        dim_key = dim
        preds_to_plot = preds[model_key][3]
    else:
        dim_key = f"output_dim_{dim}"
        preds_to_plot = preds[model_key][dim]

    train_x = influence_arrs[model_key]['train_x']
    idx = aoi_selector(train_x)
    area_influence = influence_arrs[model_key][dim_key][:, idx].sum(axis=1)  # * train_x[:].shape[0]

    scatter_with_colorbar(fig, ax[0], coords, preds_to_plot)
    ax[0].set_title(f"{model_key}: {dim_names[dim]}", )

    scatter_with_colorbar(fig, ax[1], coords, area_influence, norm=True, cmap='bwr')
    ax[1].set_title(f"Influence of area: {aoi_name} on {dim_names[dim]} (total points in area: {sum(idx)})")