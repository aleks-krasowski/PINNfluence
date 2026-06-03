"""
shall provide a fairly generic way to instantiate the problems from this repo:

https://github.com/lu-group/pinn-sampling/

which implements this paper https://www.sciencedirect.com/science/article/pii/S0045782522006260

and more :-)
"""

from pathlib import Path
from typing import Callable

import deepxde as dde
import numpy as np
import torch

from . import data
from .utils.defaults import DEFAULTS
from .utils.optimizers import SOAP
from .utils.problems import problems

DATASET_DIR = data.__path__[0]

CODNTITIONS_MAP = {
    "IC": dde.icbc.IC,
    "DirichletBC": dde.icbc.DirichletBC,
    "OperatorBC": dde.icbc.OperatorBC,
    "NeumannBC": dde.icbc.NeumannBC,
    "RobinBC": dde.icbc.RobinBC,
    "PeriodicBC": dde.icbc.PeriodicBC,
}


def create_net(
    layers=[2] + [20] * 3 + [1],
    output_transform=None,
):
    net = dde.nn.FNN(layers, "tanh", "Glorot normal")

    if output_transform is not None:
        net.apply_output_transform(output_transform)

    return net


def load_checkpoint(net, chkpt_path):
    chkpt = torch.load(chkpt_path, weights_only=False)
    print(f"Best epoch: {chkpt['epoch']}")
    net.load_state_dict(chkpt["model_state_dict"])
    return net


# all the geometries are 1D intervals over time
def create_geom(x_start=-1, x_end=1, t_start=0, t_end=1):
    geom = dde.geometry.Interval(x_start, x_end)
    timedomain = dde.geometry.TimeDomain(t_start, t_end)
    return dde.geometry.GeometryXTime(geom, timedomain)


def drop_single_point(data, drop_single_point):
    bcs_start = np.cumsum([0] + data.num_bcs)
    bcs_start = list(map(int, bcs_start))

    if drop_single_point == "none":
        return data

    elif drop_single_point == "IC":
        if isinstance(data.bcs[0], dde.icbc.IC):
            ic_points = data.train_x_bc
            ic_points = data.bcs[0].collocation_points(ic_points)
            to_drop_idx = np.random.randint(0, len(ic_points))
            point_to_drop = ic_points[to_drop_idx]

            bcs_affected = [0 for _ in range(len(data.num_bcs))]
            # check if point is included in multiple BCs
            for i in range(len(data.bcs)):
                beg, end = bcs_start[i], bcs_start[i + 1]
                if np.any((point_to_drop == data.train_x_all[beg:end]).all(axis=1)):
                    bcs_affected[i] += 1

            assert sum(bcs_affected) > 0, (
                "No BCs affected by point? This should not happen"
            )

            data.train_x_all = data.train_x_all[
                ~(data.train_x_all == point_to_drop).all(axis=1)
            ]
            # potentially also removes point from BC -- edge point
            data.train_x_bc = data.train_x_bc[
                ~(data.train_x_bc == point_to_drop).all(axis=1)
            ]
            data.train_x = data.train_x[~(data.train_x == point_to_drop).all(axis=1)]

            data.num_initial -= 1
            data.num_bcs = [bc - bcs_affected[i] for i, bc in enumerate(data.num_bcs)]

            # if affects boundary points update num_boundary
            # + 1 since we remove it bcs_affected contains one IC on which the point definitely lies
            if sum(bcs_affected) > 1:
                data.num_boundary = data.num_boundary - sum(bcs_affected) + 1

        else:
            raise ValueError(
                "drop_single_point 'IC' is only supported for IC conditions"
            )

    elif drop_single_point == "BC":
        bcs = [bc for bc in data.bcs if isinstance(bc, dde.icbc.BC)]
        if len(bcs) > 0:
            bc_points = data.train_x_bc
            bc_points = np.concatenate(
                [bc.collocation_points(bc_points) for bc in bcs], axis=0
            )
            to_drop_idx = np.random.randint(0, len(bc_points))
            point_to_drop = bc_points[to_drop_idx]

            bcs_affected = [0 for _ in range(len(data.num_bcs))]
            # check if point is included in multiple BCs
            for i in range(len(data.bcs)):
                beg, end = bcs_start[i], bcs_start[i + 1]
                if np.any((point_to_drop == data.train_x_all[beg:end]).all(axis=1)):
                    bcs_affected[i] += 1

            assert sum(bcs_affected) > 0, (
                "No BCs affected by point? This should not happen"
            )

            data.train_x_all = data.train_x_all[
                ~(data.train_x_all == point_to_drop).all(axis=1)
            ]
            data.train_x_bc = data.train_x_bc[
                ~(data.train_x_bc == point_to_drop).all(axis=1)
            ]
            data.train_x = data.train_x[~(data.train_x == point_to_drop).all(axis=1)]

            data.num_boundary -= 1
            data.num_bcs = [bc - bcs_affected[i] for i, bc in enumerate(data.num_bcs)]

            # if affects boundary points update num_boundary
            # + 1 since we remove it bcs_affected contains one IC on which the point definitely lies
            if sum(bcs_affected) > 1:
                data.num_boundary = data.num_boundary - sum(bcs_affected) + 1
        else:
            raise ValueError(
                "drop_single_point 'BC' is only supported for BC conditions"
            )
    elif drop_single_point == "domain":
        domain_points = data.train_x_all[sum(data.num_bcs) :]
        to_drop_idx = np.random.randint(0, len(domain_points))
        point_to_drop = domain_points[to_drop_idx]

        data.train_x_all = data.train_x_all[
            ~(data.train_x_all == point_to_drop).all(axis=1)
        ]
        data.train_x = data.train_x[~(data.train_x == point_to_drop).all(axis=1)]

        data.num_domain -= 1

        if data.num_domain < 0:
            raise ValueError("num_domain is negative after dropping a point")

    else:
        raise ValueError(f"drop_single_point {drop_single_point} is not supported")

    data.point_removed = point_to_drop


def create_data(
    geom,
    equation,
    num_domain=1_000,
    num_boundary=0,
    num_initial=0,
    num_test=10_000,
    solution=None,
    conditions=None,
    sample_points=None,
    drop_single_point_type="none",
):
    args = {
        "num_domain": num_domain,
        "num_boundary": num_boundary,
        "num_test": num_test,
        "solution": solution,
        "train_distribution": "Hammersley",
    }

    if sample_points is not None:
        anchors = sample_points
        args["anchors"] = anchors

    bcs = []
    if conditions is not None:
        for condition in conditions:
            # Work on a copy to avoid mutating the original dict
            cond = dict(condition)
            condition_class = CODNTITIONS_MAP[cond["target_class"]]
            if condition_class is dde.icbc.PeriodicBC:
                component_x = cond["component_x"]
                # Remove keys that are not arguments for the constructor
                cond = {
                    k: v
                    for k, v in cond.items()
                    if k not in ["target_class", "component_x"]
                }
                bcs.append(condition_class(geom, component_x, **cond))
            else:
                # Remove keys that are not arguments for the constructor
                cond = {k: v for k, v in cond.items() if k != "target_class"}
                bcs.append(condition_class(geom=geom, **cond))

    if isinstance(geom, dde.geometry.GeometryXTime):
        target_data = dde.data.TimePDE
        args["ic_bcs"] = bcs
        args["geometryxtime"] = geom
        args["pde"] = equation
        args["num_initial"] = num_initial
    else:
        target_data = dde.data.PDE
        args["bcs"] = bcs
        args["geometry"] = geom
        args["pde"] = equation

    data = target_data(**args)

    data.num_domain = num_domain

    if drop_single_point_type != "none":
        drop_single_point(data, drop_single_point_type)

    return data


def resample_validation_and_test(
    data: dde.data.Data,
    load_testdata: Callable = None,
    solution: Callable = None,
    num_validation: int = 10_000,
    num_test: int = 10_000,
):
    """
    Note that DeepXDE per default only contains a single "test" set.
    As we choose our best checkpoint based on this test set it becomes rather a validation set.
    For problems where we don't have precomputed ground truth data we thus need to sample another
    test set that is not used for checkpoint selection for final evaluation.

    in this notation:
        - test_x and test_y correspond to validation
        - holdout_test_x and holdout_test_y correspond to test
    """
    assert load_testdata is not None or solution is not None, (
        "Either load_testdata or solution must be provided"
    )

    if solution is not None:
        data.test_y = solution(data.test_x)
    else:
        assert data.test_y is None, (
            "target values were already set although no solution was provided"
        )

    # sample holdout test
    if load_testdata is not None:
        X, y = load_testdata()
        data.holdout_test_x = X
        data.holdout_test_y = y
    else:
        # get ratios of domain, initial and boundary points
        num_test_boundary = 0
        num_test_initial = 0
        num_test_domain = num_test
        if hasattr(data, "num_initial") and data.num_initial > 0:
            num_test_initial = int(num_test * (data.num_initial / (len(data.train_x))))
        if hasattr(data, "num_boundary") and data.num_boundary > 0:
            num_test_boundary = int(num_test * (data.num_boundary / len(data.train_x)))
        if hasattr(data, "num_domain") and data.num_domain > 0:
            if hasattr(data, "num_initial") and data.num_initial > 0:
                num_test_domain = int(
                    num_test
                    * (1 - (data.num_boundary + data.num_initial) / len(data.train_x))
                )
            else:
                num_test_domain = int(
                    num_test * (1 - data.num_boundary / len(data.train_x))
                )
        # Top up num_test_domain if the total is less than len(data.test_x)
        total = num_test_domain + num_test_boundary + num_test_initial
        if total < len(data.test_x):
            diff = len(data.test_x) - total
            num_test_domain += diff
        print(
            f"num_test_domain: {num_test_domain}, num_test_boundary: {num_test_boundary}, num_test_initial: {num_test_initial}"
        )

        if num_test_domain > 0:
            data.holdout_test_x = data.geom.random_points(num_test_domain, "pseudo")
        if num_test_boundary > 0:
            bc_points = data.geom.random_boundary_points(num_test_boundary, "pseudo")
            x_bcs = [bc.collocation_points(bc_points) for bc in data.bcs]
            data.holdout_test_x = np.concatenate([data.holdout_test_x] + x_bcs, axis=0)
        if num_test_initial > 0:
            ic_points = data.geom.random_initial_points(num_test_initial, "pseudo")
            x_ics = [ic.collocation_points(ic_points) for ic in data.bcs]
            data.holdout_test_x = np.concatenate([data.holdout_test_x] + x_ics, axis=0)
        print(f"data.holdout_test_x.shape: {data.holdout_test_x.shape}")
        data.holdout_test_y = solution(data.holdout_test_x)


def compile_model(
    net,
    data,
    lr=0.001,
    model=None,
    optimizer="adam",
    loss="MSE",
):
    if model is None:
        model = dde.Model(data, net)
    if optimizer == "adam":
        model.compile("adam", lr=lr, loss=loss)
    elif optimizer == "L-BFGS":
        model.compile("L-BFGS", loss=loss)
    elif optimizer == "SOAP":
        print("Using SOAP optimizer")
        optimizer_instance = SOAP(
            model.net.parameters(),
            lr=lr,
        )
        model.compile(optimizer_instance, loss=loss)
        model.opt_name = "SOAP"
    elif optimizer == "NNCG":
        model.compile("NNCG", lr=lr, loss=loss)

    return model


def restore_data(data, chkpt_path):
    chkpt = torch.load(chkpt_path, weights_only=False)

    if "train_x_all" in chkpt.keys():
        data.train_x_all = chkpt["train_x_all"]
        data.train_x = chkpt["train_x"]
        data.train_x_bc = chkpt["train_x_bc"]
        data.test_x = chkpt["test_x"]
        data.test_y = chkpt["test_y"]
        data.holdout_test_x = chkpt["holdout_test_x"]
        data.holdout_test_y = chkpt["holdout_test_y"]
        data.point_removed = chkpt.get("point_removed", None)
    return data


def construct_problem(
    problem_name: str,
    lr=0.001,
    layers=[2] + [32] * 3 + [1],
    num_domain=1_000,
    num_boundary=0,
    num_initial=0,
    optimizer="adam",
    seed=42,
    n_iterations=10_000,
    n_iterations_lbfgs=0,
    checkpoint_path=None,
    force_reinit=False,
    float64=False,
    model_version=None,
    load_path=DEFAULTS["model_zoo_src"],
    soft_constrained=True,
    drop_single_point_type="none",
) -> tuple[dde.Model, dde.data.Data, str, str]:

    if float64:
        dde.config.set_default_float("float64")

    if not soft_constrained:
        raise NotImplementedError("Only soft-constrained training is supported.")

    print(f"load_path: {load_path}")
    problem = problems[problem_name]

    equation = problem["equation"]

    output_transform = None
    conditions = problem["conditions"]

    load_testdata = problem["load_testdata"]
    solution = problem["solution"]

    if "sample_points" in problem:
        sample_points = problem["sample_points"]
    else:
        sample_points = None

    model_name = get_model_name(
        problem_name=problem_name,
        optimizer=optimizer,
        n_iterations=n_iterations,
        n_iterations_lbfgs=n_iterations_lbfgs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        layers=layers,
        seed=seed,
        float64=float64,
        soft_constrained=soft_constrained,
        point_removed=drop_single_point_type,
    )

    net = create_net(
        layers=layers,
        output_transform=output_transform,
    )

    if "feature_transform" in problem:
        net.apply_feature_transform(problem["feature_transform"])

    if "use_geometry" in problem and problem["use_geometry"]:
        geom = problem["create_geometry"]()
    else:
        geom = create_geom(
            x_start=problem.get("x_start", -1),
            x_end=problem.get("x_end", 1),
            t_start=problem.get("t_start", 0),
            t_end=problem.get("t_end", 1),
        )

    data = create_data(
        geom,
        equation,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        solution=solution,
        conditions=conditions,
        sample_points=sample_points,
        drop_single_point_type=drop_single_point_type,
    )

    resample_validation_and_test(
        data=data,
        load_testdata=load_testdata,
        solution=solution,
        num_validation=10_000,
        num_test=10_000,
    )
    print(f"Force reinit: {force_reinit}, checkpoint_path: {checkpoint_path}")
    if not force_reinit:
        if checkpoint_path is None:
            print(f"Loading checkpoint under {load_path} under name: {model_name}")
            if model_version is None or model_version == "full":
                suffix = "_full.pt"
            elif model_version == "train":
                suffix = "_train.pt"
            else:
                suffix = ".pt"

            # Try direct path first, then recursive search
            target_name = f"{model_name}{suffix}"
            direct_path = Path(load_path) / target_name

            if direct_path.exists():
                checkpoint_path = direct_path
                print(f"Found checkpoint at {checkpoint_path}")
            else:
                # Fallback to recursive search
                chkpts = list(Path(load_path).rglob(target_name))
                if len(chkpts) > 0:
                    checkpoint_path = chkpts[0]
                    print(f"Found checkpoint at {checkpoint_path}")

        if checkpoint_path is not None:
            net = load_checkpoint(net, checkpoint_path)
            data = restore_data(data, checkpoint_path)
        else:
            import warnings

            warnings.warn(
                f"No checkpoint found for '{model_name}' under '{load_path}'. "
                f"Model will use random initialization!",
                stacklevel=2,
            )

    model = compile_model(
        net,
        data,
        lr=lr,
        optimizer=optimizer,
    )

    valid_set = set(map(tuple, data.test_x))
    test_set = set(map(tuple, data.holdout_test_x))

    if len(valid_set & test_set) > 0:
        print(
            f"Test and holdout test data overlap in {len(valid_set & test_set)} points"
        )
        print(f"Test and holdout test data overlap: {valid_set & test_set}")
    return model, data, model_name, checkpoint_path


def get_model_name(
    problem_name: str,
    optimizer: str,
    n_iterations: int,
    n_iterations_lbfgs: int,
    num_domain: int,
    num_boundary: int,
    num_initial: int,
    float64: bool,
    layers=[2] + [20] * 3 + [1],
    seed=42,
    soft_constrained=True,
    point_removed="none",
):
    model_name = f"{problem_name}_{optimizer}_{n_iterations}_adam_{n_iterations_lbfgs}_lbfgs_{num_domain}_domain_{num_boundary}_boundary_{num_initial}_initial_{len(layers) - 2}_x_{layers[1]}_hidden_float64_{float64}_{seed}_{'soft' if soft_constrained else 'hard'}"
    if point_removed != "none":
        model_name += f"_point_removed_{point_removed}"
    return model_name
