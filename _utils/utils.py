import deepxde as dde
import os
import torch


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
    return os.path.join(path, files[-1])

