import captum
import deepxde as dde
import os
import time
import torch

from .dataset import DummyDataset
from .models import PINNLoss, ModelWrapper


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
        if (n_iter - self.prev_n_iter == 1) and (
            dde.optimizers.LBFGS_options["iter_per_step"] != 1
        ):
            self.model.stop_training = True
            print("Encountered broken LBFGS. Stopping training.")

        self.prev_n_iter = n_iter


def set_default_device(device: str = "cpu"):
    if device == "cpu":
        torch.set_default_device("cpu")
        print("Using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available. Using CPU")
            torch.set_default_device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch.set_default_device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS")
        else:
            print("MPS not available. Using CPU")
            torch.set_default_device("cpu")
    else:
        print("Invalid device. Using CPU")
        torch.set_default_device("cpu")


def get_checkpoint_file(path: str, prefix: str = "adam"):
    """
    Get the latest checkpoint file from a given directory.
    """
    files = os.listdir(path)
    files = [f for f in files if f.startswith(prefix)]
    files = [f for f in files if f.endswith(".pt")]
    files = sorted(files)

    if len(files) == 0:
        return None

    return os.path.join(path, files[-1])


def sample_new_training_points_via_IF(
    net: torch.nn.Module,
    data: dde.data.PDE,
    fraction_of_train: float = 5,
    batch_size: int = None,
    show_progress: bool = False,
    seed: int = None,
    num_domain: int = None,
    num_boundary: int = None,
    num_initial: int = None,
):
    """ "
    Calculate influence scores for a given model and dataset.
    Test set is sampled as a fraction of training data from the same distribution
    (hovever not from the training set itself).

    Oversample new training points to have something to choose from
    """

    if num_domain is None:
        num_domain = int(data.num_domain * fraction_of_train)
    if num_boundary is None:
        num_boundary = int(data.num_boundary * fraction_of_train)
    if num_initial is None:
        num_initial = int(data.num_initial * fraction_of_train)

    # use original training set
    train_x = data.train_x_all
    # wrap in torch dataset for DataLoader
    # return zeroes produces a dummy target tensor
    # as inside the influence function loss evaluated using
    # y_pred - y_true (where y_true is zero)
    trainset = DummyDataset(train_x, return_zeroes=True)

    pde_net = ModelWrapper(
        net=net,
        pde=data.pde,
        bcs=data.bcs,
    )

    if batch_size is None:
        batch_size = len(trainset)

    print("Approximating hessian")
    start = time.time()
    # approximate "Hessian" using training set
    if_instance = captum.influence.ArnoldiInfluenceFunction(
        pde_net,
        train_dataset=trainset,
        loss_fn=PINNLoss(),
        show_progress=show_progress,
        checkpoint="dummy",
        checkpoints_load_func=lambda x, y: 0,  # net assumed to be loaded
        batch_size=batch_size,
    )
    end = time.time()
    print(f"Approximation took: {end - start}")

    if seed is not None:
        dde.config.set_random_seed(seed)

    oversampled_data = None
    if isinstance(data, dde.data.TimePDE):
        oversampled_data = dde.data.TimePDE(
            geometryxtime=data.geom,
            pde=data.pde,
            ic_bcs=data.bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            train_distribution=data.train_distribution,
        )
    elif isinstance(data, dde.data.PDE):
        oversampled_data = dde.data.PDE(
            geometry=data.geom,
            pde=data.pde,
            bcs=data.bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=data.train_distribution,
        )
    else:
        raise NotImplementedError("Only PDE and TimePDE supported.")

    # again wrap into torch dataset
    new_trainset = DummyDataset(oversampled_data.train_x_all, return_zeroes=True)
    new_trainloader = torch.utils.data.DataLoader(
        new_trainset,
        batch_size=min(batch_size, len(new_trainset)),
    )

    # DAS WAR RECHTS
    trainloader = if_instance.train_dataloader

    # DIE NEUE RECHTE lol
    if_instance.train_dataloader = new_trainloader
    # ziemlich sicher wird der hier nicht mehr angefasst nachdem R ein mal berechnet wurde (womit H approximiert wird)
    # aber sicher ist sicher
    if_instance.hessian_dataloader = new_trainloader

    # absolut unnötig weil if not test -> model_test und loss_fn von train benutzt werden
    # und das sind die gleichen
    if_instance.model_test = pde_net
    if_instance.test_loss_fn = PINNLoss()

    # TODO: links - random gesampelte TEST punkte nicht original training AUCH BOUNDARY
    # TODO: rechts - boundary nicer samplen

    # HIER KOMMT TEST HIN - DAS LINKS☭
    print("Calculating influences")
    start = time.time()
    influences = if_instance.influence(trainloader, show_progress=show_progress)
    end = time.time()
    print(f"Influence calculation took: {end - start}")

    return influences.numpy(), oversampled_data.train_x_all, data.train_x_all
