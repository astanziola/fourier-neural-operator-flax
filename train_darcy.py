import jax
from addict import Dict
from jax import numpy as jnp
from jax import random
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from fno import FNO2D
from scripts.utils import MatlabDataset, collate_fn, log_wandb_image

# Settings dictionary
SETTINGS = Dict()
SETTINGS.data_path = 'data/darcy/darcy_211.mat'
SETTINGS.n_train = 1000
SETTINGS.n_test = 200
SETTINGS.batch_size = 20
SETTINGS.learning_rate = 0.0001 # TODO: This should be scheduled
SETTINGS.weight_decay = 1e-4
SETTINGS.n_epochs = 1000
SETTINGS.nrg = random.PRNGKey(0)

SETTINGS.fno.modes = 12
SETTINGS.fno.width = 32
SETTINGS.fno.depth = 4
SETTINGS.fno.channels_last_proj = 128
SETTINGS.fno.padding = 45

def main():
  # Loading and splitting dataset
  dataset = MatlabDataset(SETTINGS.data_path)
  train_dataset, test_dataset = random_split(
    dataset,
    [SETTINGS.n_train, SETTINGS.n_test]
  )

  # Making dataloaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=SETTINGS.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=SETTINGS.batch_size,
    shuffle=True,
    collate_fn=collate_fn
  )

  # Initialize model
  model = FNO2D(
    modes1=SETTINGS.fno.modes,
    modes2=SETTINGS.fno.modes,
    width=SETTINGS.fno.width,
    depth=SETTINGS.fno.depth,
    channels_last_proj=SETTINGS.fno.channels_last_proj,
    padding=SETTINGS.fno.padding,
  )
  _x, _ = train_dataset[0]
  _x = jnp.expand_dims(_x, axis=0)
  _, model_params = model.init_with_output(SETTINGS.nrg, _x)
  del _x

  # Initialize optimizers
  init_fun, update_fun, get_params = optimizers.adam(
    SETTINGS.learning_rate
  )
  opt_state = init_fun(model_params)

  # Define loss function
  def loss_fn(params, x, y):
    y_pred = model.apply(params, x)
    return jnp.mean(jnp.square(y - y_pred))

  @jax.jit
  def update(opt_state, x, y, step):
    params = get_params(opt_state)
    lossval, grads = jax.value_and_grad(loss_fn)(params, x, y)
    opt_state = update_fun(step, grads, opt_state)
    return opt_state, lossval

  # Initialize wandb
  print("Training...")
  wandb.init('fourier-neural-operator')

  # Training loop
  step = 0
  for epoch in range(SETTINGS.n_epochs):
    print('Epoch {}'.format(epoch))

    # Perform one epoch of training
    with tqdm(train_loader, unit="batch") as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # Update parameters
        x, y = batch
        opt_state, lossval = update(opt_state, x, y, step)

        # Log
        wandb.log({"loss": lossval}, step=step)
        tepoch.set_postfix(loss=lossval)
        step += 1

    # Get new parameters
    model_params = get_params(opt_state)

    # Log a training image
    y_pred = model.apply(model_params, x)
    log_wandb_image(wandb, "Training image", step, x[0], y[0], y_pred[0])

    # Validation
    avg_loss = 0
    val_steps = 0
    with tqdm(test_loader, unit="batch") as tval:
      for batch in tval:
        tval.set_description(f"Epoch (val) {epoch}")
        x, y = batch
        lossval = loss_fn(model_params, x, y)
        avg_loss += lossval*len(x)

        tval.set_postfix(loss=lossval)
        val_steps += 1

    wandb.log({"val_loss": lossval/val_steps}, step=step)

    # Log validation image
    _x, _y = test_dataset[0]
    _x, _y = jnp.expand_dims(_x, axis=0), jnp.expand_dims(_y, axis=0)
    _y_pred = model.apply(model_params, _x)
    log_wandb_image(wandb, "Validation image", step, _x[0], _y[0], _y_pred[0])

if __name__ == '__main__':
  main()
