import deepxde as dde 

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
        if (n_iter - self.prev_n_iter == 1) and (dde.optimizers.LBFGS_options["iter_per_step"] != 1):
            self.model.stop_training = True
            print("Encountered broken LBFGS. Stopping training.")

        self.prev_n_iter = n_iter