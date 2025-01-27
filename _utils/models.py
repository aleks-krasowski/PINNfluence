import deepxde as dde
import numpy as np
import torch

from typing import Iterable, Tuple, Literal, Callable


class ScaledFNN(dde.maps.FNN):
    """Fully-connected neural network with input scaling."""

    def __init__(self, layer_sizes, activation, kernel_initializer, L, W):
        super().__init__(layer_sizes, activation, kernel_initializer)
        self.L = L
        self.W = W

    def forward(self, inputs):
        # Assuming inputs is a tensor with shape [batch_size, 2] where
        # inputs[:, 0] corresponds to x and inputs[:, 1] corresponds to y
        x = inputs[:, 0] / self.L  # rescale x into [0, 1]
        y = inputs[:, 1] / self.W  # rescale y into [0, 1]
        scaled_inputs = torch.stack([x, y], dim=1)

        return super().forward(scaled_inputs)


class NetPredWrapper(torch.nn.Module):
    def __init__(
        self,
        net,
        pred_idx,
    ):
        super(NetPredWrapper, self).__init__()
        self.pred_idx = pred_idx
        self.net = net

    def forward(self, x):
        u = self.net(x)
        if self.pred_idx == 3:
            return torch.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2).view(-1, 1)

        if self.pred_idx == 4:
            dv_x = dde.grad.jacobian(u, x, i=1, j=0)
            du_y = dde.grad.jacobian(u, x, i=0, j=1)

            return (dv_x - du_y).view(-1, 1)

        return u[:, self.pred_idx].view(-1, 1)


class PINNLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        reduction: str = "none",
        include_all_losses: bool = True,
        include_specific_ids: int = None,
        mode: Literal["mse", "l1"] = "mse",
        weights: Iterable[int] = None,
    ):
        """
        This class assumes that inputs come from the NetworkWithPDE class.
        NetworkWithPDE returns:
            for an input Tensor of shape [batch_size, n_args]
            a Tensor of shape [batch_size, n_loss_terms]
        where n_loss_terms depends on the number of individual loss terms with
        which it was initialized.
        I.e., number of PDEs + BCs + ICs

        This classes primary intention is its use with captums influence methods.

        Its specific use case to provide different losses for training and inference in
        data attribution methods by only forwarding individual loss terms for test samples,
        and keeping all of them for training samples.

        Args:
            reduction: which method of aggregation to use across batches.
                default: none (no reduction)
            include_all_losses: whether to include all loss terms in loss computation.
                Recommended for train samples.
                default: True
            include_specific_ids: list of specific loss terms to include in loss computation.
                IDs correspond to the column index outputted by the NetworkWithPDE class.
                Recommended for test samples.
                default: None
            mode: the type of loss to use, either "mse" or "l1".
                default: "mse"
        """
        super(PINNLoss, self).__init__()
        self.reduction = reduction
        self.include_all_losses = include_all_losses
        self.include_specific_ids = include_specific_ids
        self.mode = mode.lower()
        self.weights = torch.as_tensor(weights) if weights is not None else None

        if self.mode not in ["mse", "l1"]:
            raise ValueError("Mode must be either 'mse' or 'l1'.")

        assert include_all_losses != (
            include_specific_ids is not None
        ), "include_all_losses and include_specific_ids are mutually exlusive. Please either include all losses or specify individual IDs."

    def forward(self, input, target):
        # drop unneeded predictions
        if self.include_specific_ids is not None:
            input = input[:, self.include_specific_ids]

        losses = input
        if self.mode == "mse":
            losses = losses**2
        elif self.mode == "l1":
            losses = losses.abs()

        # losses = losses.sum(axis=1)
        if self.weights is not None:
            losses *= self.weights

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "none":
            return losses
        elif self.reduction == "sum":
            return losses.sum()
        else:
            raise NotImplementedError(f"Reduction {self.reduction} not implemented")


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
        pde: Callable,
        bcs: Iterable[dde.icbc.BC] = None,
        include_pde: bool = True,
    ):
        super(ModelWrapper, self).__init__()
        self.net = net
        self.pde = pde
        self.bcs = bcs
        self.n_bcs = len(bcs) if bcs is not None else 0
        self.include_pde = include_pde

        if self.bcs is not None:
            for bc in self.bcs:
                if isinstance(bc, dde.icbc.initial_conditions.IC):
                    bc.on_boundary = bc.on_initial

        assert (
            include_pde or bcs is not None
        ), "At least one of PDEs or BCs must be included."

        self.net.eval()

    def forward(self, x):
        x_np = x.detach().cpu().numpy()

        torch.autograd.set_grad_enabled(True)
        if not x.requires_grad:
            x = x.requires_grad_(True)
        outputs = self.net(x)

        # Handle PDE outputs as list
        losses = []
        if self.include_pde:
            f = self.pde(x, outputs)
            if not isinstance(f, (list, tuple)):
                f = [f]
            losses.extend([fi.view(-1, 1) for fi in f])

        # Handle boundary conditions
        if self.bcs is not None:
            for bc in self.bcs:
                bc_loss = torch.zeros((x.shape[0], 1), device=x.device)
                bc_mask = torch.tensor(bc.on_boundary(x_np, np.ones_like(x_np[:, 0])))

                if bc_mask.any():
                    x_subset = x[bc_mask].clone().detach().requires_grad_(True)
                    outputs_subset = self.net(x_subset)

                    # NOTE:
                    # in NeumannBC and RobinBC
                    # the first argument (X) is incorrectly handled by deepxde if gradient is enabled
                    # make the following adjustments in deepxde/icbc/boundary_conditions.py
                    """
                    def normal_derivative(self, X, inputs, outputs, beg, end):
                        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
                        if backend_name == 'pytorch' and X.requires_grad:
                            n = self.boundary_normal(X.clone().detach().numpy(), beg, end, None)
                        else:
                            n = self.boundary_normal(X, beg, end, None)
                        return bkd.sum(dydx * n, 1, keepdims=True)
                    """
                    # NOTE: there may be a cleaner solution - yet to be found
                    bc_loss_curr = bc.error(
                        x_subset,
                        x_subset,
                        outputs_subset,
                        0,
                        bc_mask.sum(),
                    )
                    bc_loss[bc_mask] = bc_loss_curr

                losses.append(bc_loss)

        # Stack all losses
        total_loss = torch.cat(losses, dim=1)
        dde.grad.clear()
        return total_loss
