import os
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore[assignment]

# Default fallback path; callers should pass log_dir to LocalLogger instead.
LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self, log_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.log_path = Path(log_dir) if log_dir is not None else LOG_PATH
        self.writer = SummaryWriter(log_dir=str(
            self.log_path / "tensorboard")) if SummaryWriter is not None else None

    @property
    def experiment(self):
        return self.writer

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.writer is None:
            return
        for key, value in dict(metrics or {}).items():
            scalar = None
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalar = float(value)
            elif torch.is_tensor(value) and value.numel() == 1:
                scalar = float(value.detach().cpu().item())
            if scalar is not None:
                self.writer.add_scalar(key, scalar, step)

    @rank_zero_only
    def log_video(
        self,
        key: str,
        video: Any,
        step: Optional[int] = None,
        fps: int = 30,
        **kwargs,
    ):
        assert step is not None
        if self.writer is not None:
            try:
                tensor = video
                if isinstance(tensor, list):
                    tensor = torch.stack([v if torch.is_tensor(
                        v) else torch.as_tensor(v) for v in tensor])
                if torch.is_tensor(tensor):
                    if tensor.ndim == 4:
                        tensor = tensor.unsqueeze(0)
                    self.writer.add_video(
                        key, tensor.detach().cpu(), step, fps=fps)
            except Exception:
                pass

        dir = self.log_path / key
        dir.mkdir(exist_ok=True, parents=True)
        if isinstance(video, list):
            video = video[0]
        if torch.is_tensor(video):
            video = video.detach().cpu()
        np.save(dir / f"{step:0>6}.npy", np.asarray(video))

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                Image.fromarray(image_np).save(path)
                if self.writer is not None:
                    self.writer.add_image(
                        key, image.detach().cpu(), step, dataformats="CHW")
            else:
                image_np = np.asarray(image)
                Image.fromarray(image_np).save(path)
                if self.writer is not None:
                    if image_np.ndim == 3 and image_np.shape[-1] in (1, 3, 4):
                        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                        self.writer.add_image(key, tensor, step)

    def finalize(self, status):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
