from typing import Callable, Iterable, Literal, Tuple

import deepxde as dde
import numpy as np
import torch


class PINNLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        reduction: str = "none",
        include_all_losses: bool = True,
        include_specific_ids: int = None,
        mode: Literal["mse", "l1", "l2"] = "mse",
        weights: Iterable[int] = None,
    ):
        """
        This class assumes that inputs come from the ModelWrapper class.
        ModelWrapper returns:
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
                IDs correspond to the column index outputted by the ModelWrapper class.
                Recommended for test samples.
                default: None
            mode: the type of loss to use, either "mse" (equivalent to "l2") or "l1".
                default: "mse"
        """
        super(PINNLoss, self).__init__()
        self.reduction = reduction
        self.include_all_losses = include_all_losses
        self.include_specific_ids = include_specific_ids
        self.mode = mode.lower()
        self.weights = torch.as_tensor(weights) if weights is not None else None

        if self.mode not in ["mse", "l2", "l1"]:
            raise ValueError("Mode must be either 'mse' or 'l1'.")

        assert include_all_losses != (include_specific_ids is not None), (
            "include_all_losses and include_specific_ids are mutually exlusive. Please either include all losses or specify individual IDs."
        )

    def forward(self, input, target):
        # drop unneeded predictions
        if self.include_specific_ids is not None:
            input = input[:, self.include_specific_ids]

        losses = input
        if self.mode == "mse" or self.mode == "l2":
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
        self.default_float = dde.config.default_float()

        if self.bcs is not None:
            for bc in self.bcs:
                if isinstance(bc, dde.icbc.initial_conditions.IC):
                    bc.on_boundary = bc.on_initial

        assert include_pde or bcs is not None, (
            "At least one of PDEs or BCs must be included."
        )

        self.net.eval()

    def forward(self, x):
        self.net.zero_grad()

        if self.default_float == "float64" and not isinstance(x, torch.DoubleTensor):
            x = x.double()

        outputs = self.net(x)
        # to ensure all parameters are in the graph
        zero_term = outputs.sum() * 0

        # Handle PDE outputs as list
        losses = []
        if self.include_pde:
            f = self.pde(x, outputs)
            if not isinstance(f, (list, tuple)):
                f = [f]
            losses.extend([fi.view(-1, 1) for fi in f])

        # Handle boundary conditions
        if self.bcs is not None:
            x_np = x.detach().cpu().numpy().astype(self.default_float)
            for i, bc in enumerate(self.bcs):
                bc_loss = torch.zeros((x.shape[0], 1), device=x.device)

                if not isinstance(bc, dde.icbc.PeriodicBC):
                    if isinstance(bc, dde.icbc.IC):
                        bc_mask = torch.tensor(
                            bc.on_initial(x_np, bc.geom.on_initial(x_np))
                        ).bool()
                    else:
                        bc_mask = torch.tensor(
                            bc.on_boundary(x_np, bc.geom.on_boundary(x_np))
                        ).bool()

                    if bc_mask.any():
                        x_subset = x[bc_mask].clone().detach().requires_grad_(True)
                        outputs_subset = self.net(x_subset)

                        # Usual handling
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
                        # NOTE: there may be a cleaner solution - yet to be found:
                        bc_loss_curr = bc.error(
                            x_subset.detach().numpy(),
                            x_subset,
                            outputs_subset,
                            0,
                            bc_mask.sum(),
                        )

                        bc_loss[bc_mask] = bc_loss_curr

                else:
                    bc_mask = torch.tensor(
                        bc.on_boundary(x_np, bc.geom.on_boundary(x_np))
                    ).bool()

                    bc_points = bc.collocation_points(x_np)
                    bc_points_tensor = torch.tensor(
                        bc_points, dtype=x.dtype, device=x.device, requires_grad=True
                    )
                    bc_points_outputs = self.net(bc_points_tensor)

                    bc_loss_curr = bc.error(
                        bc_points_tensor.detach().numpy(),
                        bc_points_tensor,
                        bc_points_outputs,
                        0,
                        len(bc_points),
                    )

                    bc_loss[bc_mask] = bc_loss_curr

                losses.append(bc_loss)

        # ensure all parameters are in the graph
        losses = [loss + zero_term for loss in losses]
        # Stack all losses
        total_loss = torch.cat(losses, dim=1)
        dde.grad.clear()
        return total_loss


# class ModelWrapper(torch.nn.Module):
#     def __init__(
#         self,
#         net: torch.nn.Module,
#         data: dde.data.Data,
#         bcs: Iterable[dde.icbc.BC] = None,
#         include_pde: bool = True,
#     ):
#         """
#         Args:
#             net: The neural network model to be wrapped.
#             data: The data object to be used.
#             bcs: The boundary conditions to be used.
#             include_pde: Whether to include the PDEs in the loss.
#             pure_dataloss_indices: The boundary condition indices that are used solely for data loss.
#                                    This will ignore all other PDE and BC calculations for identified points.
#         """
#         super(ModelWrapper, self).__init__()
#         self.net = net
#         self.data = data
#         self.bcs = bcs
#         self.n_bcs = len(bcs) if bcs is not None else 0
#         self.include_pde = include_pde

#         # Static cache (for full-batch, fixed-point training)
#         self._static_enabled = False
#         self._static_cache = None

#         self.total_loss_terms, self.total_pde_terms = self._get_total_loss_terms_count()

#         # 1) define tolerance FIRST
#         self._match_tol = 1e-12

#         # 2) precompute data keys
#         data_all = self._canonicalize_points(self.data.train_x_all)
#         self._data_keys_sorted = self._unique_rows(data_all)

#         # 3) precompute union of PointSetBC points
#         bc_keys_list = []
#         self._pointset_cache = {}
#         if self.bcs is not None:
#             for bc in self.bcs:
#                 if isinstance(bc, dde.icbc.PointSetBC):
#                     canonical = self._canonicalize_points(bc.points)
#                     bc_keys_list.append(canonical)
#                     self._pointset_cache[id(bc)] = self._build_pointset_cache_entry(
#                         bc, canonical
#                     )
#         if bc_keys_list:
#             bc_stack = torch.cat(bc_keys_list, dim=0)
#             self._bc_union_keys_sorted = self._unique_rows(bc_stack)
#         else:
#             self._bc_union_keys_sorted = torch.empty(
#                 (0, data_all.shape[1]), dtype=torch.int64
#             )

#         # 4) now it’s safe to compute pure_dataloss_indices
#         pure_dataloss_indices = {}
#         if bcs is not None:
#             tensor_column_counter = self.total_pde_terms
#             for i, bc in enumerate(bcs):
#                 if isinstance(bc, dde.icbc.PointSetBC):
#                     # identify if THIS BC has any points that are in BCs but NOT in data
#                     # (since union is already built, _identify_pure_dataloss_points works now)
#                     if self._identify_pure_dataloss_points(bc.points, bcs).any():
#                         component = bc.component
#                         if isinstance(component, numbers.Number):
#                             pure_dataloss_indices[i] = [tensor_column_counter]
#                             tensor_column_counter += 1
#                         elif isinstance(component, Iterable):
#                             pure_dataloss_indices[i] = [
#                                 tensor_column_counter + j for j in range(len(component))
#                             ]
#                             tensor_column_counter += len(component)
#                         else:
#                             raise ValueError(
#                                 f"Invalid component type in PointSetBC at index {i}: {type(component)}"
#                             )

#         self.pure_dataloss_indices = pure_dataloss_indices

#         assert include_pde or bcs is not None, (
#             "At least one of PDEs or BCs must be included."
#         )

#         self.net.eval()
#         self.net.zero_grad()

#     def __len__(self):
#         return 1

#     def __iter__(self):
#         yield self

#     def set_static_points(self, x):
#         """Enable cached masks/indices for fixed full-batch training.

#         Call once with the tensor you will repeatedly train on (no resampling).
#         This avoids per-iteration canonicalization, membership checks, and
#         boundary masking. If called again, the cache is rebuilt.
#         """
#         x_tensor, x_np = self._prepare_x(x)
#         self._static_cache = self._build_static_cache(x_tensor, x_np)
#         self._static_enabled = True

#     def _build_static_cache(self, x_tensor, x_np):
#         device = x_tensor.device
#         dtype = x_tensor.dtype

#         canonical = self._canonicalize_points(x_np)
#         bc_mask_cpu = self._membership_mask(canonical, self._bc_union_keys_sorted)
#         data_mask_cpu = self._membership_mask(canonical, self._data_keys_sorted)
#         pure_dataloss_mask = (bc_mask_cpu & ~data_mask_cpu).to(device=device)

#         bc_masks = []
#         pointset_indices = {}
#         periodic_masks = {}
#         ic_masks = {}
#         generic_masks = {}

#         if self.bcs is not None:
#             for bc in self.bcs:
#                 if isinstance(bc, dde.icbc.PointSetBC):
#                     cache = self._pointset_cache.get(id(bc))
#                     if cache is None:
#                         canonical_bc = self._canonicalize_points(bc.points)
#                         cache = self._build_pointset_cache_entry(bc, canonical_bc)
#                         self._pointset_cache[id(bc)] = cache

#                     lookup = cache["lookup"]
#                     x_indices = []
#                     bc_indices = []
#                     for idx, row in enumerate(canonical.tolist()):
#                         bc_idx = lookup.get(tuple(row))
#                         if bc_idx is not None:
#                             x_indices.append(idx)
#                             bc_indices.append(bc_idx)

#                     if x_indices:
#                         pointset_indices[id(bc)] = (
#                             torch.tensor(x_indices, device=device, dtype=torch.long),
#                             torch.tensor(bc_indices, device=device, dtype=torch.long),
#                         )
#                     else:
#                         pointset_indices[id(bc)] = None
#                     bc_masks.append(None)  # handled via indices

#                 elif isinstance(bc, dde.icbc.PeriodicBC):
#                     bc_mask_raw = bc.on_boundary(x_np, bc.geom.on_boundary(x_np))
#                     bc_mask = self._ensure_tensor(
#                         bc_mask_raw, device=device, dtype=torch.bool
#                     )
#                     periodic_masks[id(bc)] = bc_mask
#                     bc_masks.append(bc_mask)

#                 elif isinstance(bc, dde.icbc.IC):
#                     ic_mask = self._ensure_tensor(
#                         bc.on_initial(x_np, bc.geom.on_initial(x_np)),
#                         device=device,
#                         dtype=torch.bool,
#                     )
#                     ic_masks[id(bc)] = ic_mask
#                     bc_masks.append(ic_mask)

#                 elif isinstance(bc, dde.icbc.BC):
#                     bc_mask_raw = bc.on_boundary(x_np, bc.geom.on_boundary(x_np))
#                     bc_mask = self._ensure_tensor(
#                         bc_mask_raw, device=device, dtype=torch.bool
#                     )
#                     generic_masks[id(bc)] = bc_mask
#                     bc_masks.append(bc_mask)
#                 else:
#                     bc_masks.append(None)

#         return {
#             "canonical": canonical,
#             "pure_dataloss_mask": pure_dataloss_mask,
#             "pointset_indices": pointset_indices,
#             "periodic_masks": periodic_masks,
#             "ic_masks": ic_masks,
#             "generic_masks": generic_masks,
#         }

#     def forward(self, *args, **kwargs):
#         # Accept (x) or (x, y, ...) and ignore labels/aux
#         if len(args) == 0:
#             raise TypeError(
#                 "ModelWrapper.forward expects at least one positional argument (x)."
#             )
#         x = args[0]
#         x, x_np = self._prepare_x(x)

#         # Determine if we can use the static cache
#         # Cache is only valid if it was built for the same number of points as the current batch
#         cache = None
#         if self._static_enabled and self._static_cache is not None:
#             # Check if cache size matches current batch size
#             cache_size = len(self._static_cache.get("canonical", []))
#             if cache_size == x.shape[0]:
#                 cache = self._static_cache
#             # else: cache doesn't match batch size, compute masks from tensors

#         if len(self.pure_dataloss_indices) > 0:
#             return self._forward_with_pure_dataloss(x, x_np, cache)
#         else:
#             return self._forward(x, x_np, cache)

#     def _forward_with_pure_dataloss(self, x, x_np, cache=None):
#         if cache is not None:
#             pure_dataloss_mask = cache["pure_dataloss_mask"]
#         else:
#             pure_dataloss_mask = self._identify_pure_dataloss_points(x_np, self.bcs)
#         non_dataloss_mask = ~pure_dataloss_mask

#         # Compute losses for non-dataloss points (if any exist)
#         if non_dataloss_mask.any():
#             non_dataloss_losses = self._forward(
#                 x[non_dataloss_mask], x_np[non_dataloss_mask], cache
#             )
#         else:
#             # If no non-dataloss points, create empty tensor with correct shape
#             non_dataloss_losses = torch.empty(
#                 (0, self.total_loss_terms), device=x.device, dtype=x.dtype
#             )

#         # Compute losses for pure dataloss points
#         if pure_dataloss_mask.any():
#             dataloss_losses_list = [
#                 self.handle_pointsetbc_points(
#                     x[pure_dataloss_mask],
#                     x_np[pure_dataloss_mask],
#                     self.bcs[i],
#                     outputs=None,
#                 )
#                 for i in self.pure_dataloss_indices.keys()
#             ]

#             with torch.enable_grad():
#                 dataloss_out = self.net(x[pure_dataloss_mask])
#                 dataloss_zero_term = torch.nn.MSELoss()(
#                     dataloss_out[0] * 0, torch.zeros_like(dataloss_out[0])
#                 )

#             dataloss_losses = torch.zeros(
#                 (pure_dataloss_mask.sum(), self.total_loss_terms),
#                 device=x.device,
#                 dtype=x.dtype,
#             )

#             # Calculate the correct column indices for each pure dataloss BC
#             for e, (bc_idx, column_indices) in enumerate(
#                 self.pure_dataloss_indices.items()
#             ):
#                 bc_loss = dataloss_losses_list[e] + dataloss_zero_term
#                 dataloss_losses[:, column_indices] = bc_loss
#         else:
#             # If no pure dataloss points, create empty tensor
#             dataloss_losses = torch.empty(
#                 (0, self.total_loss_terms), device=x.device, dtype=x.dtype
#             )

#         # Combine results
#         total_loss = torch.empty(
#             (x.shape[0], self.total_loss_terms), device=x.device, dtype=x.dtype
#         )

#         if non_dataloss_mask.any():
#             total_loss[non_dataloss_mask] = non_dataloss_losses
#         if pure_dataloss_mask.any():
#             total_loss[pure_dataloss_mask] = dataloss_losses

#         return total_loss

#     @torch.no_grad()
#     def _get_total_loss_terms_count(self):
#         """Calculate the total number of loss terms without performing a forward pass."""
#         total_terms = 0
#         pde_terms = 0

#         # Count PDE loss terms
#         if self.include_pde:
#             # We need to determine how many PDE equations there are
#             # This is tricky without a forward pass, so we'll use a dummy forward pass
#             # with a single point to determine the structure
#             dummy_x = torch.zeros(
#                 (1, self.data.geom.dim),
#                 device=next(self.net.parameters()).device,
#                 requires_grad=True,
#             )
#             with torch.enable_grad():
#                 dummy_outputs = self.net(dummy_x)
#                 f = self.data.pde(dummy_x, dummy_outputs)
#                 if not isinstance(f, (list, tuple)):
#                     f = [f]
#                 total_terms += len(f)
#                 pde_terms += len(f)

#         # Count boundary condition loss terms
#         if self.bcs is not None:
#             for bc in self.bcs:
#                 if isinstance(bc, dde.icbc.PointSetBC):
#                     # For PointSetBC, we need to check the component specification
#                     component = bc.component
#                     if isinstance(component, numbers.Number):
#                         total_terms += 1
#                     elif isinstance(component, Iterable):
#                         total_terms += len(component)
#                     else:
#                         # If component is None or not specified, we need to determine from bc_values
#                         if hasattr(bc.values, "shape") and len(bc.values.shape) > 1:
#                             total_terms += bc.values.shape[1]
#                         else:
#                             # Fallback: use a dummy forward pass to determine the actual number of components
#                             dummy_x = torch.zeros(
#                                 (1, self.data.geom.dim),
#                                 device=next(self.net.parameters()).device,
#                                 requires_grad=True,
#                             )
#                             with torch.enable_grad():
#                                 dummy_outputs = self.net(dummy_x)
#                                 # Create dummy bc_values with the same shape as network output
#                                 dummy_bc_values = torch.zeros_like(dummy_outputs)
#                                 computed_loss = dummy_outputs - dummy_bc_values
#                                 if computed_loss.ndim == 2:
#                                     total_terms += computed_loss.shape[1]
#                                 else:
#                                     total_terms += 1
#                 else:
#                     # For other BC types, assume 1 loss term per BC
#                     total_terms += 1

#         return total_terms, pde_terms

#     def _ensure_tensor(
#         self,
#         value,
#         *,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> torch.Tensor:
#         tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
#         if dtype is not None and tensor.dtype != dtype:
#             tensor = tensor.to(dtype)
#         if device is not None:
#             device = torch.device(device)
#             if tensor.device != device:
#                 tensor = tensor.to(device)
#         return tensor

#     def _canonicalize_points(self, value) -> torch.Tensor:
#         """Quantize coordinates to an integer grid for stable equality checks."""
#         tensor = self._ensure_tensor(value, device="cpu", dtype=torch.float32)
#         if tensor.ndim != 2:
#             raise ValueError(f"Expected 2D coordinates, got shape {tensor.shape}.")
#         if self._match_tol is not None:
#             tensor = torch.round(tensor / self._match_tol)
#         return tensor.to(torch.int32)

#     def _unique_rows(self, tensor: torch.Tensor) -> torch.Tensor:
#         if tensor.numel() == 0:
#             return tensor
#         uniques = torch.unique(tensor, dim=0)
#         return uniques

#     def _membership_mask(
#         self, keys: torch.Tensor, uniques: torch.Tensor
#     ) -> torch.Tensor:
#         if uniques.numel() == 0:
#             return torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device)
#         keys_expanded = keys.unsqueeze(1)
#         uniques_expanded = uniques.unsqueeze(0)
#         # Matching on canonicalized integer grid avoids float drift.
#         matches = keys_expanded.eq(uniques_expanded).all(dim=2)
#         return matches.any(dim=1)

#     def _build_pointset_cache_entry(
#         self, bc: dde.icbc.PointSetBC, canonical: torch.Tensor
#     ) -> dict:
#         """
#         Pre-compute tensor caches for PointSetBC to avoid NumPy conversions during forward passes.

#         NOTE(deepxde-compat): DeepXDE should expose tensor-friendly buffers long-term.
#         """
#         canonical_cpu = canonical.clone()
#         lookup = {}
#         for idx, row in enumerate(canonical_cpu.tolist()):
#             lookup[tuple(row)] = idx

#         bc_points = self._ensure_tensor(bc.points, device="cpu", dtype=torch.float64)
#         bc_values = self._ensure_tensor(bc.values, device="cpu", dtype=torch.float64)
#         if bc_values.ndim == 1:
#             bc_values = bc_values.unsqueeze(1)
#         return {
#             "points": bc_points,
#             "values": bc_values,
#             "canonical": canonical_cpu,
#             "lookup": lookup,
#         }

#     def _identify_pure_dataloss_points(self, x_np, bcs):
#         # bcs is unused now; we precomputed the union in __init__
#         x_keys = self._canonicalize_points(x_np)

#         bc_mask_cpu = self._membership_mask(x_keys, self._bc_union_keys_sorted)
#         data_mask_cpu = self._membership_mask(x_keys, self._data_keys_sorted)

#         mask_cpu = bc_mask_cpu & ~data_mask_cpu
#         target_device = x_np.device if torch.is_tensor(x_np) else torch.device("cpu")
#         return mask_cpu.to(device=target_device)

#     def _prepare_x(self, x):
#         # If upstream passes (x, y, ...), take the first element as inputs
#         if isinstance(x, (tuple, list)):
#             x = x[0]
#         p = next(self.net.parameters())
#         if torch.is_tensor(x):
#             x_tensor = x
#         elif hasattr(x, "__array__"):
#             x_tensor = torch.as_tensor(x)
#         else:
#             raise TypeError("x must be array-like or torch.Tensor")

#         if x_tensor.device != p.device or x_tensor.dtype != p.dtype:
#             x_tensor = x_tensor.to(device=p.device, dtype=p.dtype)

#         # Handle case when inside torch.func transforms (vjp, vmap, etc.)
#         # where tensors don't have storage
#         try:
#             x_np = x_tensor.detach().cpu().numpy()
#         except RuntimeError:
#             x_np = None  # Inside functional transform - must use cache

#         return x_tensor, x_np

#     def _forward(self, x, x_np, cache=None):
#         # Ensure x has gradients enabled for Jacobian computations in BCs
#         if not x.requires_grad:
#             x = x.requires_grad_(True)

#         with torch.enable_grad():
#             outputs = self.net(x)

#         # zero term ensures all model parameters are in the graph
#         zero_term = outputs.sum() * 0
#         losses = []

#         # Handle PDE outputs as list
#         if self.include_pde:
#             f = self.data.pde(x, outputs)
#             if not isinstance(f, (list, tuple)):
#                 f = [f]
#             losses.extend([fi.sum(axis=1).view(-1, 1) + zero_term for fi in f])

#         # Handle boundary conditions
#         if self.bcs is not None:
#             for bc in self.bcs:
#                 cached_mask = None
#                 cached_indices = None
#                 if cache is not None:
#                     if isinstance(bc, dde.icbc.PointSetBC):
#                         cached_indices = cache["pointset_indices"].get(id(bc))
#                     elif isinstance(bc, dde.icbc.IC):
#                         cached_mask = cache["ic_masks"].get(id(bc))
#                     elif isinstance(bc, dde.icbc.PeriodicBC):
#                         cached_mask = cache["periodic_masks"].get(id(bc))
#                     elif isinstance(bc, dde.icbc.BC):
#                         cached_mask = cache["generic_masks"].get(id(bc))

#                 if isinstance(bc, dde.icbc.IC):
#                     bc_loss = self.handle_ic_points(x, x_np, bc, outputs, cached_mask)
#                 elif isinstance(bc, dde.icbc.PointSetBC):
#                     bc_loss = self.handle_pointsetbc_points(
#                         x, x_np, bc, outputs, cached_indices
#                     )
#                 elif isinstance(bc, dde.icbc.PeriodicBC):
#                     bc_loss = self.handle_periodicbc_points(
#                         x, x_np, bc, cached_mask
#                     )
#                 elif isinstance(bc, dde.icbc.BC):
#                     # NOTE:
#                     # in NeumannBC and RobinBC
#                     # the first argument (X) is incorrectly handled by deepxde if gradient is enabled
#                     # make the following adjustments in deepxde/icbc/boundary_conditions.py
#                     """
#                     def normal_derivative(self, X, inputs, outputs, beg, end):
#                         dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
#                         if backend_name == 'pytorch' and bkd.is_tensor(X) and X.requires_grad:
#                             n = self.boundary_normal(
#                                 X.clone().detach().numpy(), beg, end, None)
#                         else:
#                             n = self.boundary_normal(X, beg, end, None)
#                         return bkd.sum(dydx * n, 1, keepdims=True)
#                     """

#                     bc_loss = self.handle_generic_bc_points(
#                         x, x_np, bc, outputs, cached_mask
#                     )
#                 else:
#                     raise ValueError(f"Unknown BC type: {type(bc)}")

#                 if bc_loss.shape[1] == 1:
#                     losses.append(bc_loss.view(-1, 1) + zero_term)
#                 elif bc_loss.shape[1] > 1:
#                     for i in range(bc_loss.shape[1]):
#                         losses.append(bc_loss[:, i].view(-1, 1) + zero_term)
#                 else:
#                     raise ValueError(
#                         f"Invalid number of dimensions for bc_loss: {bc_loss.ndim}"
#                     )

#         # Stack all losses
#         total_loss = torch.cat(losses, dim=1)
#         dde.grad.clear()
#         return total_loss

#     def handle_ic_points(self, x, x_np, ic, outputs, mask=None):
#         ic_loss_curr = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

#         ic_mask = mask
#         if ic_mask is None:
#             ic_mask = torch.tensor(
#                 ic.on_initial(x, ic.geom.on_initial(x)),
#                 device=x.device,
#                 dtype=torch.bool,
#             )

#         # if there are any points, compute the loss
#         if ic_mask.any():
#             x_subset = x[ic_mask].clone()
#             outputs_subset = outputs[ic_mask]
#             ic_loss_curr[ic_mask] = ic.error(
#                 # x_subset.detach().cpu().numpy(),
#                 x_subset,
#                 x_subset,
#                 outputs_subset,
#                 0,
#                 ic_mask.sum(),
#             )

#         return ic_loss_curr

#     def handle_pointsetbc_points(self, x, x_np, pointsetbc, outputs, cached_indices=None):
#         if outputs is None:
#             with torch.enable_grad():
#                 outputs = self.net(x)

#         component = pointsetbc.component

#         # Determine the correct number of components for the loss tensor
#         if isinstance(component, numbers.Number):
#             num_components = 1
#         elif isinstance(component, Iterable):
#             num_components = len(component)
#         else:
#             raise ValueError(f"Invalid component type in PointSetBC: {type(component)}")

#         bc_loss_curr = torch.zeros(
#             (x.shape[0], num_components), device=x.device, dtype=x.dtype
#         )

#         # Find exact matches between x_np and cached BC points using integer grid hashing
#         cache = self._pointset_cache.get(id(pointsetbc))
#         if cache is None:
#             # NOTE(deepxde-compat): fallback for BCs created after ModelWrapper init.
#             canonical = self._canonicalize_points(pointsetbc.points)
#             cache = self._build_pointset_cache_entry(pointsetbc, canonical)
#             self._pointset_cache[id(pointsetbc)] = cache

#         if cached_indices is not None:
#             cached_pair = cached_indices
#         else:
#             bc_lookup = cache["lookup"]
#             x_keys = self._canonicalize_points(x)

#             x_indices = []
#             bc_indices = []
#             for idx, row in enumerate(x_keys):
#                 bc_idx = bc_lookup.get(tuple(row))
#                 if bc_idx is not None:
#                     x_indices.append(idx)
#                     bc_indices.append(bc_idx)

#             if len(x_indices) > 0:
#                 cached_pair = (
#                     torch.tensor(x_indices, device=x.device, dtype=torch.long),
#                     torch.tensor(bc_indices, device=x.device, dtype=torch.long),
#                 )
#             else:
#                 cached_pair = None

#         if cached_pair is not None:
#             x_indices_tensor, bc_indices_tensor = cached_pair

#             bc_values_stored = cache["values"]
#             bc_values_subset = bc_values_stored.to(device=x.device, dtype=x.dtype)[
#                 bc_indices_tensor
#             ]
#             if bc_values_subset.ndim == 1:
#                 bc_values_subset = bc_values_subset.unsqueeze(1)

#             if bc_values_subset.shape[0] != cache["values"].shape[0]:
#                 warnings.warn(
#                     (
#                         f"Number of points in PointSetBC ({bc_values_subset.shape[0]}) does not match number of values ({cache['values'].shape[0]}).\n"
#                         "This is either caused by batching or a mismatch in the number of points and values.\n"
#                         "Consider setting anchors in the data class."
#                     ),
#                     category=Warning,
#                 )

#             # Compute network outputs for matched points
#             bc_points_outputs = outputs[x_indices_tensor]

#             # Compute loss based on component specification
#             if isinstance(component, numbers.Number):
#                 computed_loss = (
#                     bc_points_outputs[:, component : component + 1]
#                     - bc_values_subset[:, component : component + 1]
#                 )
#             elif isinstance(component, Iterable):
#                 computed_loss = (
#                     bc_points_outputs[:, component] - bc_values_subset[:, component]
#                 )
#             else:
#                 computed_loss = bc_points_outputs - bc_values_subset

#             # Assign losses to corresponding positions in bc_loss_curr
#             if computed_loss.ndim == 1:
#                 computed_loss = computed_loss.view(-1, 1)
#                 bc_loss_curr[x_indices_tensor] = computed_loss.view(-1, 1)
#             elif computed_loss.ndim == 2:
#                 bc_loss_curr = torch.zeros(
#                     (x.shape[0], computed_loss.shape[1]), device=x.device, dtype=x.dtype
#                 )
#                 bc_loss_curr[x_indices_tensor] = computed_loss
#             else:
#                 raise ValueError(
#                     f"Invalid number of dimensions for computed_loss: {computed_loss.ndim}"
#                 )

#         else:
#             warnings.warn(
#                 (
#                     f"No points to compute loss for PointSetBC found.\n"
#                     "This is either caused by batching or a mismatch in the number of points and values.\n"
#                     "Consider settings anchors in the data class."
#                 ),
#                 category=Warning,
#             )

#         return bc_loss_curr

#     def handle_periodicbc_points(self, x, x_np, periodicbc, mask=None):
#         bc_loss_curr = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

#         if mask is None:
#             bc_mask_raw = periodicbc.on_boundary(x_np, periodicbc.geom.on_boundary(x_np))
#             bc_mask = self._ensure_tensor(bc_mask_raw, device=x.device, dtype=torch.bool)
#         else:
#             bc_mask = mask

#         if bc_mask.any():
#             # generate periodic boundary points
#             bc_points = periodicbc.collocation_points(x_np)
#             bc_points_tensor = self._ensure_tensor(
#                 bc_points,
#                 device=x.device,
#                 dtype=x.dtype,
#             ).requires_grad_(True)
#             bc_points_outputs = self.net(bc_points_tensor)

#             bc_loss_curr[bc_mask] = periodicbc.error(
#                 # bc_points_tensor.detach().cpu().numpy(),
#                 bc_points_tensor,
#                 bc_points_tensor,
#                 bc_points_outputs,
#                 0,
#                 len(bc_points),
#             )

#         return bc_loss_curr

#     def handle_generic_bc_points(self, x, x_np, bc, outputs, mask=None):
#         bc_loss_curr = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

#         if mask is None:
#             bc_mask_raw = bc.on_boundary(x, bc.geom.on_boundary(x))
#             # NOTE(deepxde-compat): expects DeepXDE boundary checks to accept torch tensors.
#             bc_mask = self._ensure_tensor(bc_mask_raw, device=x.device, dtype=torch.bool)
#         else:
#             bc_mask = mask

#         if bc_mask.any():
#             # Don't clone - we need to maintain gradient connections
#             x_subset = x[bc_mask]
#             outputs_subset = outputs[bc_mask]

#             # For OperatorBC, we need to recompute outputs with gradients enabled
#             # to ensure the computational graph is properly connected
#             if isinstance(bc, dde.icbc.OperatorBC):
#                 with torch.enable_grad():
#                     x_subset_grad = (
#                         x_subset.requires_grad_(True)
#                         if not x_subset.requires_grad
#                         else x_subset
#                     )
#                     outputs_subset_grad = self.net(x_subset_grad)
#                     bc_error = bc.error(
#                         x_subset_grad,
#                         x_subset_grad,
#                         outputs_subset_grad,
#                         0,
#                         bc_mask.sum(),
#                     )
#             else:
#                 bc_error = bc.error(
#                     x_subset,
#                     x_subset,
#                     outputs_subset,
#                     0,
#                     bc_mask.sum(),
#                 )

#             bc_loss_curr[bc_mask] = bc_error

#         return bc_loss_curr


class NetPredWrapper(torch.nn.Module):
    def __init__(
        self,
        net,
        pred_idx=None,
    ):
        super(NetPredWrapper, self).__init__()
        self.pred_idx = pred_idx
        self.net = net
        self.default_float = dde.config.default_float()

    def forward(self, x):
        if self.default_float == "float64" and not isinstance(x, torch.DoubleTensor):
            x = x.double()

        u = self.net(x)

        if self.pred_idx is not None and u.shape[1] > 1:
            u = u[:, self.pred_idx]
        return u.view(-1, 1)


class ModifiedMLP(dde.nn.NN):
    """
    Modified MLP with gated mixing:
        u = act(W_u x_in + b_u)
        v = act(W_v x_in + b_v)
        for each hidden layer:
            x = act(W x + b)
            x = x * u + (1 - x) * v
        y = W_out x + b_out

    Args:
        layer_sizes: [in_dim, hidden_dim, ..., hidden_dim, out_dim].
                     All hidden dims must be equal (like the JAX impl).
        activation:  name or callable, e.g. "tanh" (single activation for all places).
        kernel_initializer: name understood by deepxde.initializers.get (e.g., "glorot_normal").
        regularization: optional, kept for API parity with FNN.
    """

    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=None
    ):
        super().__init__()

        if len(layer_sizes) < 3:
            raise ValueError(
                "layer_sizes must be at least [in_dim, hidden_dim, out_dim]."
            )

        hidden_dims = layer_sizes[1:-1]
        if len(set(hidden_dims)) != 1:
            raise ValueError(
                "All hidden layers must have the same width for ModifiedMLP."
            )
        self.in_dim = layer_sizes[0]
        self.hidden_dim = hidden_dims[0]
        self.out_dim = layer_sizes[-1]
        self.num_layers = len(hidden_dims)

        # Activation: JAX version uses a single activation everywhere.
        if isinstance(activation, list):
            if len(activation) != 1:
                raise ValueError("ModifiedMLP expects a single activation function.")
            self.activation = dde.nn.activations.get(activation[0])
        else:
            self.activation = dde.nn.activations.get(activation)

        init_w = dde.nn.initializers.get(kernel_initializer)
        init_b = dde.nn.initializers.get("zeros")
        self.regularizer = regularization

        # Gating projections u, v : R^{in_dim} -> R^{hidden_dim}
        self.linear_u = torch.nn.Linear(
            self.in_dim, self.hidden_dim, dtype=dde.config.real(torch)
        )
        self.linear_v = torch.nn.Linear(
            self.in_dim, self.hidden_dim, dtype=dde.config.real(torch)
        )
        init_w(self.linear_u.weight)
        init_b(self.linear_u.bias)
        init_w(self.linear_v.weight)
        init_b(self.linear_v.bias)

        # Main hidden stack: first layer maps in_dim -> hidden_dim; the rest are hidden_dim -> hidden_dim
        self.linears = torch.nn.ModuleList()
        self.linears.append(
            torch.nn.Linear(self.in_dim, self.hidden_dim, dtype=dde.config.real(torch))
        )
        init_w(self.linears[-1].weight)
        init_b(self.linears[-1].bias)
        for _ in range(self.num_layers - 1):
            self.linears.append(
                torch.nn.Linear(
                    self.hidden_dim, self.hidden_dim, dtype=dde.config.real(torch)
                )
            )
            init_w(self.linears[-1].weight)
            init_b(self.linears[-1].bias)

        # Output layer: hidden_dim -> out_dim
        self.out_linear = torch.nn.Linear(
            self.hidden_dim, self.out_dim, dtype=dde.config.real(torch)
        )
        init_w(self.out_linear.weight)
        init_b(self.out_linear.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Compute gates from the original (possibly transformed) input
        u = self.activation(self.linear_u(x))
        v = self.activation(self.linear_v(x))

        # Layer loop with gated mixing
        for linear in self.linears:
            x = self.activation(linear(x))
            x = x * u + (1.0 - x) * v

        # Final projection
        x = self.out_linear(x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
