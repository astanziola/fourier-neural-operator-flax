from turtle import width
from typing import Callable

import flax.linen as nn
from flax.linen.initializers import normal
from jax import numpy as jnp


class SpectralConv2d(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12

  @nn.compact
  def __call__(self, x):
    # x.shape: [batch, height, width, in_channels]

    # Initialize parameters
    in_channels = x.shape[-1]
    scale = 1/(in_channels * self.out_channels)
    in_channels = x.shape[-1]
    height = x.shape[1]
    width = x.shape[2]

    # Checking that the modes are not more than the input size
    assert self.modes1 <= height//2 + 1
    assert self.modes2 <= width//2 + 1
    assert height % 2 == 0 # Only tested for even-sized inputs
    assert width % 2 == 0 # Only tested for even-sized inputs

    # The model assumes real inputs and therefore uses a real
    # fft. For a 2D signal, the conjugate symmetry of the
    # transform is exploited to reduce the number of operations.
    # Given an input signal of dimesions (N, C, H, W), the
    # output signal will have dimensions (N, C, H, W//2+1).
    # Therefore the kernel weigths will have different dimensions
    # for the two axis.
    kernel = self.param(
      'kernel',
      normal(scale, jnp.complex64),
      (in_channels, self.out_channels, 2*self.modes1+1, self.modes2),
      jnp.complex64
    )

    # Perform fft of the input
    x_ft = jnp.fft.rfftn(x, axes=(1, 2))

    # Shift the zero frequency to the center of the array, only for the
    # first axis.
    x_ft = jnp.fft.fftshift(x_ft, axes=1)

    # Get the center of the spectrum
    center_idx = x_ft.shape[1]//2
    center_spect = x_ft[:, -self.modes1+center_idx:self.modes1+center_idx+1, :self.modes2,:]

    # Multiply the center of the spectrum by the kernel
    x_filt = jnp.einsum('bijc,coij->bijo', center_spect, kernel)

    # Pad the kernel with zeros to restore the original size
    pad_size_1 = (x_ft.shape[1] - 2*self.modes1)//2
    pad_size_2 = x_ft.shape[2] - self.modes2
    x_filt = jnp.pad(
      x_filt,
      ((0, 0), (pad_size_1, pad_size_1-1), (0, pad_size_2), (0, 0)),
      mode='constant'
    )

    # Restore the zero frequency to the beginning of the array
    x_filt = jnp.fft.ifftshift(x_filt, axes=1)

    # Go back to the spatial domain
    y = jnp.fft.irfftn(x_filt, axes=(1, 2))

    return y

class FourierStage(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, x):
    x_fourier = SpectralConv2d(
      out_channels=self.out_channels,
      modes1=self.modes1,
      modes2=self.modes2
    )(x)
    x_local = nn.Conv(
      self.out_channels,
      (1,1),
    )(x)
    return self.activation(x_fourier + x_local)


class FNO2D(nn.Module):
  r'''
  Fourier Neural Operator for 2D signals.

  Implemented from
  https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

  Attributes:
    modes1: Number of modes in the first dimension.
    modes2: Number of modes in the second dimension.
    width: Number of channels to which the input is lifted.
    depth: Number of Fourier stages
    channels_last_proj: Number of channels in the hidden layer of the last
      2-layers Fully Connected (channel-wise) network
    activation: Activation function to use
    out_channels: Number of output channels, >1 for non-scalar fields.
  '''
  modes1: int = 12
  modes2: int = 12
  width: int = 32
  depth: int = 4
  channels_last_proj: int = 128
  activation: Callable = nn.gelu
  out_channels: int = 1
  padding: int = 0 # Padding for non-periodic inputs

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # Generate coordinate grid, and append to input channels
    grid = self.get_grid(x)
    x = jnp.concatenate([x, grid], axis=-1)

    # Lift the input to a higher dimension
    x = nn.Dense(self.width)(x)

    # Pad input
    if self.padding > 0:
      x = jnp.pad(
        x,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant'
      )

    # Apply Fourier stages, last one has no activation
    # (can't find this in the paper, but is in the original code)
    for depthnum in range(self.depth):
      activation = self.activation if depthnum < self.depth - 1 else lambda x: x
      x = FourierStage(
        out_channels=self.width,
        modes1=self.modes1,
        modes2=self.modes2,
        activation=activation
      )(x)

    # Unpad
    if self.padding > 0:
      x = x[:, :-self.padding, :-self.padding, :]

    # Project to the output channels
    x = nn.Dense(self.channels_last_proj)(x)
    x = self.activation(x)
    x = nn.Dense(self.out_channels)(x)

    return x


  @staticmethod
  def get_grid(x):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid
