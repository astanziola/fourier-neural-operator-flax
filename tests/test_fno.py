from jax import numpy as jnp
from jax.random import PRNGKey

from fno import FNO2D


def test_fno():
  rng = PRNGKey(0)
  inputs = jnp.ones((8, 256, 256, 1))
  output, variables = FNO2D().init_with_output(rng, inputs)
  print(output.shape)
  print(output.dtype)
  assert output.shape == (8, 256, 256, 1)

  # Test forward pass
  output = FNO2D().apply(variables, inputs)

if __name__ == '__main__':
  test_fno()
