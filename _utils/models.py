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

class NetworkWithPDE(torch.nn.Module):
    def __init__(
            self,
            pinn: torch.nn.Module,
            pdes: Iterable[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            bcs: Iterable[Tuple[dde.icbc.BC, Callable[[torch.Tensor], torch.Tensor]]] = None,
            ics: Iterable[Tuple[dde.icbc.IC, Callable[[torch.Tensor], torch.Tensor]]] = None,
            geom: dde.geometry.Geometry = None
    ):
        """

        Args:
            pinn: physics informed neural network
            pdes: partial derivative functions
            bcs: boundary conditions and on_boundary checkers
            ics: initial conditions and on_boundary checkers
        """
        super(NetworkWithPDE, self).__init__()
        self.pinn = pinn
        self.pdes = pdes
        self.geom = geom
        self.bcs = bcs
        self.ics = ics
        self.n_pdes = len(pdes) if pdes is not None else 0
        self.n_bcs = len(bcs) if bcs is not None else 0
        self.n_ics = len(ics) if ics is not None else 0
        self.n_loss_terms = self.n_pdes + self.n_bcs + self.n_ics

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        # similarityInfluence disables this for whatever reason
        # better safe than sorry
        torch.autograd.set_grad_enabled(True)
        if not x.requires_grad:
            x = x.requires_grad_(True)
        u = self.pinn(x)

        total_loss = torch.zeros((u.shape[0], self.n_loss_terms))

        if self.pdes is not None:
            for e, pde in enumerate(self.pdes):
                pde_out = pde(x, u).sum(axis=1).view(-1,1)
                total_loss[:, e:e+1] += pde_out

        if self.bcs is not None:
            for e, (bc, on_boundary) in enumerate(self.bcs, start=self.n_pdes):
                bc_loss = torch.zeros((u.shape[0],1))
                bc_subset = on_boundary(x).bool() & torch.from_numpy(self.geom.on_boundary(x_np)).bool().to(x.device)

                if bc_subset.any():
                    if bc_subset.all():
                        x_subset = x
                        u_subset = u
                    else:
                        x_subset = x[bc_subset].clone().detach().requires_grad_(True)
                        u_subset = self.pinn(x_subset)

                    bc_loss_curr = bc.error(
                        x_subset,  # X
                        x_subset,  # inputs
                        u_subset,  # outputs
                        0,  # start_idx
                        bc_subset.sum()  # end_idx
                    )
                    bc_loss[bc_subset] += bc_loss_curr

                total_loss[:, e:e+1] += bc_loss

        if self.ics is not None:
            for e, (ic, on_boundary) in enumerate(self.ics, self.n_pdes + self.n_bcs):
                ic_loss = torch.zeros((u.shape[0], 1))
                ic_subset = on_boundary(x)  & torch.from_numpy(self.geom.on_boundary(x_np)).bool().to(x.device)

                if ic_subset.any():
                    if ic_subset.all():
                        x_subset = x
                        u_subset = u
                    else:
                        x_subset = x[ic_subset].clone().detach().requires_grad_(True)
                        u_subset = self.pinn(x_subset)

                    ic_loss_curr = ic.error(
                        x_subset,  # X
                        x_subset,  # inputs
                        u_subset,  # outputs
                        0,  # start_idx
                        ic_subset.sum()  # end_idx
                    )
                    ic_loss[ic_subset] += ic_loss_curr

                total_loss[:, e:e+1] += ic_loss
        # dde will cache all gradients filling up memory very quickly
        dde.grad.clear()
        return total_loss
