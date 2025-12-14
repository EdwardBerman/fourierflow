import os
import pickle
from pathlib import Path
from typing import Optional, Union
from os import PathLike

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

_PATH = Union[str, Path, PathLike]

from .callback import Callback


def _logger_name(logger) -> str:
    # Lightning loggers usually have `.name`; otherwise fall back to class name.
    return getattr(logger, "name", None) or logger.__class__.__name__


def _logger_version(logger) -> str:
    # Lightning loggers usually have `.version`; otherwise fall back to 0.
    v = getattr(logger, "version", None)
    if v is None:
        return "0"
    return str(v)


class JAXModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, mode):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.dirpath = None

        assert mode in ["min", "max"]
        self.best_metric = float("inf") if mode == "min" else -float("inf")

    def on_validation_epoch_end(self, trainer, routine):
        self.__resolve_ckpt_dir(trainer)

        metric = trainer.logs[self.monitor]
        stats = f"{self.monitor}={metric:.4f}"
        path = self.dirpath / f"epoch={trainer.current_epoch}-{stats}.ckpt"

        if self.is_better(metric):
            self.best_metric = metric

            for old_path in list(self.dirpath.glob("*.ckpt")):
                old_path.unlink()

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(routine.params, f)

    def is_better(self, metric):
        if self.mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def __resolve_ckpt_dir(self, trainer) -> None:
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )
            ckpt_path = trainer.weights_save_path / "checkpoints" / version
        else:
            ckpt_path = trainer.weights_save_path / "checkpoints"

        self.dirpath = ckpt_path


class CustomModelCheckpoint(ModelCheckpoint):
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        self.__resolve_ckpt_dir(trainer)
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)

        # NOTE: setting these attributes needs to happen as early as possible BEFORE reloading callback states,
        # because the attributes are part of the state_key which needs to be fully defined before reloading.
        if self._save_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._save_on_train_epoch_end = (
                trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1
            )

    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> None:
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        if self.dirpath is not None:
            return  # short circuit

        # Safe default
        save_dir = trainer.default_root_dir

        # Lightning gives either `trainer.logger` (single) or `trainer.loggers` (list)
        loggers = getattr(trainer, "loggers", None) or []
        if not loggers and getattr(trainer, "logger", None) is not None:
            loggers = [trainer.logger]

        # If there is exactly one logger and it has a save_dir, prefer it
        if len(loggers) == 1 and getattr(loggers[0], "save_dir", None) is not None:
            save_dir = loggers[0].save_dir or trainer.default_root_dir

        # Robust version string (no private trainer attrs, no hidden helpers)
        version = None
        if len(loggers) == 1:
            version = getattr(loggers[0], "version", None)

        if version is None:
            version = "version_0"
        elif not isinstance(version, str):
            version = f"version_{version}"

        ckpt_path = os.path.join(save_dir, "checkpoints", version)
        ckpt_path = trainer.strategy.broadcast(ckpt_path)

        self.dirpath = ckpt_path

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        if self.save_top_k != 0 and self._fs.isdir(dirpath) and len(self._fs.ls(dirpath)) > 0:
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
