import time
from typing import Optional

import captum
import deepxde as dde
import numpy as np
import torch

from .dataset import DummyDataset
from .models import ModelWrapper, PINNLoss

def instantiate_IF(
    model,
    model_name: str,
    batch_size: int = None,
    show_progress: bool = False,
    seed: int = 0,
    use_train_set: bool = True,
    tmp_dir: Optional[str] = "tmp/r_cache/",
    prefer_load_R: bool = True,
    projection_dim: int = 50,
):
    """Create an instance of Influence Function estimator"""
    data = model.data
    net = model.net

    # Sample points for hessian approximation
    if use_train_set:
        influences_set = data.train_x_all
    else:
        influences_set = data.geom.random_points(1_000, random="pseudo")

    print(f"Influences set shape (right side of equation): {influences_set.shape}")
    # Add dummy zero targets for DataLoader
    influences_set = DummyDataset(influences_set, return_zeroes=True)

    if batch_size is None:
        batch_size = min(1024, len(influences_set))

    pde_net = ModelWrapper(
        net=net,
        pde=data.pde,
        bcs=data.bcs,
    )

    print("Approximating hessian")
    start = time.time()
    # Approximate Hessian using Arnoldi method
    if_instance = captum.influence.ArnoldiInfluenceFunctionPrecomputed(
        pde_net,
        train_dataset=influences_set,
        loss_fn=PINNLoss(),
        show_progress=show_progress,
        checkpoint="dummy",
        checkpoints_load_func=lambda x, y: 0,  # net assumed to be loaded
        batch_size=batch_size,
        seed=seed,
        r_cache_path=f"{tmp_dir}/{model_name}_r_cache.pt",
        prefer_load_R=prefer_load_R,
        projection_dim=projection_dim,
    )
    end = time.time()
    print(f"Approximation took: {end - start}")

    return if_instance


def instantiate_grad_dot(
    model,
    batch_size: int = None,
    use_train_set: bool = True,
    show_progress: bool = False,
):
    """Create an instance of gradient-dot product estimator"""
    data = model.data
    net = model.net

    pde_net = ModelWrapper(
        net,
        pde=data.pde,
        bcs=data.bcs,
    )

    if batch_size is None:
        batch_size = len(data.train_x_all)

    if use_train_set:
        print("Using training set for grad dot")
        trainset = DummyDataset(data.train_x_all, return_zeroes=True)
    else:
        print("Using random points for grad dot")
        trainset = DummyDataset(
            data.geom.random_points(1_000, random="pseudo"), return_zeroes=True
        )

    graddot = captum.influence.TracInCP(
        model=pde_net,
        loss_fn=PINNLoss(),
        train_dataset=trainset,
        checkpoints=["dummy"],
        # Set checkpoint contribution to 1 to get grad dot product
        checkpoints_load_func=lambda x, y: 1,
        batch_size=batch_size,
    )

    return graddot


def calculate_influence_scores(
    candidate_points,
    tda_instance: captum.influence.ArnoldiInfluenceFunction | captum.influence.TracInCP,
    show_progress: bool = False,
    batch_size: int = None,
):
    """Calculate influence scores for candidate points"""
    # Use appropriate batch size
    if batch_size is None:
        batch_size = min(1024, len(candidate_points))
    else:
        batch_size = min(batch_size, len(candidate_points))

    candidate_set = DummyDataset(candidate_points, return_zeroes=True)
    candidate_loader = torch.utils.data.DataLoader(
        candidate_set,
        batch_size=batch_size,
    )

    test_samples = candidate_loader

    print("Calculating influences")
    start = time.time()
    influences = tda_instance.influence(test_samples, show_progress=show_progress)
    end = time.time()
    print(f"Influence calculation took: {end - start}")

    return influences.detach().numpy().astype(np.float32)


def sample_random_points(
    geometry: dde.geometry.Geometry,
    num_points: int = 1,
    num_bcs: Optional[int] = None,
):
    """Sample random points from the geometry maintaining the same ratios as model data.

    Args:
        geometry: The geometry to sample from
        num_points: Total number of points to sample
    """

    if hasattr(geometry, "random_boundary_points"):
        if hasattr(geometry, "random_initial_points") and num_bcs is not None:
            num_domain = int(9 * num_points / 10)
            num_remaining = num_points - num_domain
            num_initial = int(num_remaining / num_bcs)
            num_boundary = num_points - num_domain - num_initial

            points = geometry.random_points(num_domain, random="pseudo")
            points_initial = geometry.random_initial_points(
                num_initial, random="pseudo"
            )
            points_boundary = geometry.random_boundary_points(
                num_boundary, random="pseudo"
            )
            points = np.vstack([points, points_initial, points_boundary])
        else:
            points = geometry.random_points((9 * num_points // 10), random="pseudo")
            points_boundary = geometry.random_boundary_points(
                (num_points // 10), random="pseudo"
            )
            points = np.vstack([points, points_boundary])
    else:
        points = geometry.random_points(num_points, random="pseudo")

    return points
