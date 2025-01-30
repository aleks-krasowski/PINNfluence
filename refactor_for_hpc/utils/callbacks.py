from deepxde.callbacks import Callback
import torch


class BestModelCheckpoint(Callback):
    """Save the PyTorch model after every epoch, with the annoying appending of DeepXDE.

    Args:
        filepath (str): Path to save the model file.
        verbose (int): Verbosity mode, 0 or 1.
        save_better_only (bool): If True, only saves the model if the monitored
            metric improves.
        monitor (str): The metric to monitor ('train_loss' or 'val_loss').
        min_delta (float): Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        monitor="test loss",
        min_delta=0.0,
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = torch.inf

    def on_train_begin(self):
        self.best = self.get_monitor_value()
        self._save_model(0, self.best)

    def on_epoch_end(self):
        """Save the model at the end of an epoch."""
        current = self.get_monitor_value()
        epoch = self.model.train_state.epoch

        if current is None:
            raise ValueError(f"Metric '{self.monitor}' is not available in logs.")

        if self.save_better_only:
            if current < self.best - self.min_delta:
                self.best = current
                self._save_model(epoch, current)
        else:
            self._save_model(epoch, current)

    def _save_model(self, epoch, current):
        """Save the model to the specified filepath."""
        if self.verbose > 0:
            print(
                f"Epoch {epoch}: {self.monitor} improved to {current:.2e}, saving model to {self.filepath}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.net.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
        }
        torch.save(checkpoint, self.filepath)

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result
