from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset


def gaussian_normalize(
  x: np.ndarray,
  eps=0.00001
) -> Tuple[np.ndarray, float, float]:
  '''Adapted from
  https://github.com/zongyi-li/fourier_neural_operator/blob/c13b475dcc9bcd855d959851104b770bbcdd7c79/utilities3.py#L73

  Attributes:
    x (np.ndarray): Input array
    eps (float): Small number to avoid division by zero

  Returns:
    Tuple[np.ndarray, float, float]: Normalized array, mean and standard deviation
  '''
  mean = np.mean(x, 0, keepdims=True)
  std = np.std(x, 0, keepdims=True)
  x = (x - mean) / (std + eps)
  return x, mean, std

class MatlabDataset(Dataset):
  def __init__(self, path: str):
    # Load matfile
    mat = loadmat(path)

    # Extract data and put batch dimension in front
    self.x = mat['inputs'].astype(np.float32)
    self.y = mat['outputs'].astype(np.float32)
    self.x = np.moveaxis(self.x, -1, 0)
    self.y = np.moveaxis(self.y, -1, 0)

    # Add channel dimension at the end
    self.x = self.x[..., np.newaxis]
    self.y = self.y[..., np.newaxis]

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

def collate_fn(batch):
  x, y = zip(*batch)
  x = np.stack(x, axis=0)
  y = np.stack(y, axis=0)
  return x, y

def log_wandb_image(
  wandb,
  name: str,
  step: int,
  x: np.ndarray,
  y: np.ndarray,
  y_pred: np.ndarray
) -> None:
  fig, ax = plt.subplots(1, 3, figsize=(12, 4))

  ax[0].imshow(x, cmap="inferno")
  ax[0].set_title("Input map")

  ax[1].imshow(y, cmap="inferno")
  ax[1].set_title("Target field")

  ax[2].imshow(y_pred, cmap="inferno")
  ax[2].set_title("Predicted field")

  img = wandb.Image(plt)
  wandb.log({name: img}, step=step)
  plt.close()
