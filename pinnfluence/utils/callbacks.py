import os
import time
from collections.abc import Iterable
from typing import Callable

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from deepxde.callbacks import Callback
from deepxde.metrics import l2_relative_error, mean_squared_error
from tqdm import tqdm

from .utils import plot_prediction_heatmap


class BestModelCheckpoint(Callback):
    """
    Callback to save the best model based on a specified metric.
    """

    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        restore_best=True,
        monitor="test loss",
        min_delta=0.0,
        period=100,
        total_epochs=None,
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.restore_best = restore_best
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = torch.inf
        self.best_epoch = 0
        self.start_time = None
        self.period = period
        self.total_epochs = total_epochs
        self.pbar = None

    def on_train_begin(self):
        self.best = self.get_monitor_value()
        self._save_model(0, self.best)
        self.start_time = time.time()

        if self.verbose > 0 and self.total_epochs is not None:
            self.pbar = tqdm(
                total=self.total_epochs,
                desc="Training",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

    def on_epoch_end(self):
        """Save the model at the end of an epoch."""
        if self.model.train_state.epoch % self.period == 0:
            current = self.get_monitor_value()
            epoch = self.model.train_state.epoch

            if current is None:
                raise ValueError(f"Metric '{self.monitor}' is not available in logs.")

            if self.save_better_only:
                if current < (self.best - self.min_delta):
                    self.best = current
                    self._save_model(epoch, current)
            else:
                self._save_model(epoch, current)

            if self.pbar is not None:
                self.pbar.set_postfix(
                    {
                        "epoch": epoch,
                        f"{self.monitor}": f"{current:.2e}",
                        "best": f"{self.best:.2e}",
                    }
                )
                self.pbar.update(self.period)

    def on_train_end(self):
        if self.pbar is not None:
            self.pbar.close()

        if self.verbose > 0:
            print(
                f"Training completed. Best {self.monitor} = {self.best:.2e} @ {self.best_epoch}"
            )
            print(f"Training time: {time.time() - self.start_time:.2f}s")

        if self.restore_best:
            checkpoint = torch.load(self.filepath, weights_only=False)
            self.model.net.load_state_dict(checkpoint["model_state_dict"])

    def _save_model(self, epoch, current, filepath=None):
        """Save the model to the specified filepath."""
        if filepath is None:
            filepath = self.filepath

        if self.verbose > 1:
            print(
                f"Epoch {epoch}: {self.monitor} improved to {current:.2e}, saving model to {self.filepath}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.net.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
            "train_x_all": self.model.data.train_x_all,
            "train_x": self.model.data.train_x,
            "train_x_bc": self.model.data.train_x_bc,
            "test_x": self.model.data.test_x,
            "test_y": self.model.data.test_y,
            "holdout_test_x": self.model.data.holdout_test_x,
            "holdout_test_y": self.model.data.holdout_test_y,
            "point_removed": getattr(self.model.data, "point_removed", None),
            "epoch": epoch,
        }
        torch.save(checkpoint, filepath)

        self.best_epoch = epoch

    def get_monitor_value(self):
        if self.monitor == "train loss":
            # Handle both single loss and iterable of losses
            train_loss = self.model.train_state.loss_train
            result = (
                sum(train_loss)
                if isinstance(train_loss, Iterable)
                and not isinstance(train_loss, (str, bytes))
                else train_loss
            )
        elif self.monitor == "test loss":
            # Handle both single loss and iterable of losses
            test_loss = self.model.train_state.loss_test
            result = (
                sum(test_loss)
                if isinstance(test_loss, Iterable)
                and not isinstance(test_loss, (str, bytes))
                else test_loss
            )
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class EvalMetricCallback(Callback):
    """
    Evaluate the model on the test data after every epoch.
    """

    def __init__(self, verbose=0, filepath=None, verbose_period=1, total_epochs=None):
        super().__init__()
        self.verbose = verbose
        self.last_test_loss_state = None
        self.filepath = filepath
        self.verbose_period = verbose_period
        self.total_epochs = total_epochs
        self.file = None
        self.pbar = None

    def on_train_begin(self):
        self.X = self.model.data.holdout_test_x
        self.y = self.model.data.holdout_test_y

        if self.filepath is not None:
            file_exists = os.path.exists(self.filepath)
            self.file = open(self.filepath, "a")
            if not file_exists:
                self.file.write(
                    "epoch,train_loss,valid_loss,test_loss,l2_relative_error,mse,mean_residual,mean_abs_residual,optimizer_name\n"
                )

        y_pred = self.model.predict(self.X)
        l2re = l2_relative_error(self.y, y_pred)
        self.model.train_state.l2re = l2re
        mse = mean_squared_error(self.y, y_pred)
        self.model.train_state.mse = mse

        optimizer_name = self.model.opt_name

        row = (0, np.nan, np.nan, np.nan, l2re, mse, optimizer_name)

        if self.filepath is not None:
            self.file.write(",".join(map(str, row)) + "\n")

        if self.verbose == 1 and self.total_epochs is not None:
            self.pbar = tqdm(
                total=self.total_epochs,
                desc="Training",
                # bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

    def on_epoch_end(self):
        if any(self.model.train_state.loss_test != self.last_test_loss_state):
            self.last_test_loss_state = self.model.train_state.loss_test

            epoch = self.model.train_state.epoch
            train_loss = self.model.train_state.loss_train
            # Handle both single loss and iterable of losses
            train_loss = (
                sum(train_loss)
                if isinstance(train_loss, Iterable)
                and not isinstance(train_loss, (str, bytes))
                else train_loss
            )

            valid_loss = self.model.train_state.loss_test
            # Handle both single loss and iterable of losses
            valid_loss = (
                sum(valid_loss)
                if isinstance(valid_loss, Iterable)
                and not isinstance(valid_loss, (str, bytes))
                else valid_loss
            )

            test_loss_pde = np.mean(
                np.square(self.model.predict(self.X, operator=self.model.data.pde))
            )

            y_pred = self.model.predict(self.X)
            l2re = l2_relative_error(self.y, y_pred)
            self.model.train_state.l2re = l2re
            mse = mean_squared_error(self.y, y_pred)
            self.model.train_state.mse = mse

            optimizer_name = self.model.opt_name

            row = (
                epoch,
                train_loss,
                valid_loss,
                test_loss_pde,
                l2re,
                mse,
                optimizer_name,
            )

            if self.filepath is not None:
                self.file.write(",".join(map(str, row)) + "\n")

            if self.pbar is not None and (epoch % self.verbose_period) == 0:
                self.pbar.set_postfix(
                    {
                        # 'epoch': epoch,
                        "train_loss": f"{train_loss:.2e}",
                        "valid_loss": f"{valid_loss:.2e}",
                        "test_loss": f"{test_loss_pde:.2e}",
                        "l2re": f"{l2re:.2e}",
                        "mse": f"{mse:.2e}",
                    }
                )
                self.pbar.update(epoch - self.pbar.n)

            self.model.loss_history = dde.model.LossHistory()

    def on_train_end(self):
        if self.pbar is not None:
            self.pbar.close()

        if self.file is not None:
            self.file.close()


class WandbCallback(dde.callbacks.Callback):
    """Callback to log metrics to Wandb during training.

    Attributes:
        project (str, default: 'pinnfluence'): Wandb project name.
        name (str, optional): Wandb experiment name. If left empty a random name will be assigned.
        period (int, default: 1): Interval (number of epochs) between logging metrics.
    """

    def __init__(self, project: str = "pinnfluence", name: str = None, period: int = 1):
        super().__init__()

        if not wandb.api.api_key:
            raise EnvironmentError("Wandb API key not found.")

        self.project = project
        self.name = name
        self.period = period
        self.run = None
        self.epochs_since_last = 0
        self.cur_epoch = 0
        self.last_epoch = 0
        self.global_epoch = 0

    def init(self):
        """Initialize the Wandb run."""
        if self.run is None:
            self.run = wandb.init(project=self.project, name=self.name)


class WandbCallbackLoss(WandbCallback):
    """Callback to log metrics to Wandb during training.

    Attributes:
        project (str, default: 'thesis'): Wandb project name.
        name (str, optional): Wandb experiment name. If left empty a random name will be assigned.
        period (int, default: 1): Interval (number of epochs) between logging metrics.
                                Note: calls model._test() each time adding overhead
        loss_names (list, optional): List of loss names to log.
    """

    def __init__(
        self,
        project: str = "thesis",
        name: str = None,
        period: int = 50,
        loss_names: Iterable[str] = None,
    ):
        super().__init__(project, name, period)
        self.loss_names = loss_names

    def log_losses(self):
        old_rank = dde.config.rank
        # set to something != 0 to prevent dde from printing while calling _test
        dde.config.rank = 1
        self.model._test()
        dde.config.rank = old_rank

        epoch = self.global_epoch + self.model.train_state.epoch
        loss_train = self.model.train_state.loss_train
        loss_test = self.model.train_state.loss_test

        y_pred = self.model.predict(self.model.data.holdout_test_x)
        l2re = l2_relative_error(self.model.data.holdout_test_y, y_pred)

        wandb.log(
            {
                "Loss/train_total": np.sum(loss_train),
                "Loss/test_total": np.sum(loss_test),
                "L2RE": l2re,
                "epoch": epoch,
            },
            step=epoch,
        )

        # if not set then init as "loss_1", "loss_2", ...
        if self.loss_names is None:
            self.loss_names = [f"{i}" for i in range(len(loss_train))]
        for e, (_loss_train, _loss_test) in enumerate(zip(loss_train, loss_test)):
            wandb.log(
                {
                    f"Loss/train_{self.loss_names[e]}": _loss_train,
                    f"Loss/test_{self.loss_names[e]}": _loss_test,
                    "epoch": epoch,
                },
                step=epoch,
            )

    def on_train_begin(self):
        if self.global_epoch == 0:
            self.log_losses()

    def on_epoch_end(self):
        """Log metrics at the end of each epoch."""
        self.cur_epoch = self.model.train_state.epoch
        self.epochs_since_last = self.cur_epoch - self.last_epoch
        if self.epochs_since_last >= self.period:
            self.log_losses()
            self.last_epoch = self.cur_epoch

    def on_train_end(self):
        self.cur_epoch = self.model.train_state.epoch
        self.epochs_since_last = self.cur_epoch - self.last_epoch
        if self.epochs_since_last > 0:
            self.log_losses()

        self.global_epoch += self.cur_epoch
        self.last_epoch = 0


class WandbCallbackPlots(WandbCallback):
    """Callback to log metrics to Wandb during training.

    Attributes:
        project (str, default: 'thesis'): Wandb project name.
        name (str, optional): Wandb experiment name. If left empty a random name will be assigned.
        period (int, default: 500): Interval (number of epochs) between logging metrics.
                                    Note: Plotting is fairly expensive in both runtime and memory.
                                    Thus choose this carefully.
        plotting_func: (Callable): Plotting function. Expected to take y_true, y_pred, residuals, title as arguments
    """

    def __init__(
        self,
        project: str = "thesis",
        name: str = None,
        period: int = 500,
        plotting_func: Callable[
            [np.ndarray, np.ndarray, np.ndarray, str, str], plt.Figure
        ] = plot_prediction_heatmap,
    ):
        super().__init__(project, name, period)
        self.plotting_func = plotting_func
        if self.plotting_func is None:
            print(
                "no plotting function supplied - will attempt to use the default DeepXDE plotting function"
            )

    def plot(self):
        epoch = self.global_epoch + self.cur_epoch

        y_true = self.model.data.holdout_test_y
        y_pred = self.model.predict(self.model.data.holdout_test_x)
        residuals = self.model.predict(
            self.model.data.holdout_test_x, operator=self.model.data.pde
        )
        title = f"Epoch {epoch}"

        fig = self.plotting_func(
            self.model.data.holdout_test_x, y_true, y_pred, residuals, title
        )

        wandb.log({"prediction figure": wandb.Image(fig)}, step=epoch)
        plt.close()

    def on_train_begin(self):
        if self.global_epoch == 0:
            self.plot()

    def on_epoch_end(self):
        self.cur_epoch = self.model.train_state.epoch
        self.epochs_since_last = self.cur_epoch - self.last_epoch
        if self.epochs_since_last >= self.period:
            self.plot()
            self.last_epoch = self.cur_epoch

    def on_train_end(self):
        self.cur_epoch = self.model.train_state.epoch
        self.epochs_since_last = self.cur_epoch - self.last_epoch
        if self.epochs_since_last > 0:
            self.plot()

        self.global_epoch += self.cur_epoch
        self.last_epoch = 0
