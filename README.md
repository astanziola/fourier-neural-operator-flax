# Fourier Neural Operator (Flax)

This is an **unofficial** [Flax]() ([JAX]()) port of the [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) developed by Zongyi Li et al.

Please visit [this repository](https://github.com/zongyi-li/fourier_neural_operator) to access the official PyTorch implementation of FNO form the paper's authors, and for the citation policy in case you use FNO in your research.

## Requirements

To install `fno`, you just need to have `flax` installed (see the `requirements.txt` file).

Then:
- To generate the data you need `MATLAB` (or, probably, Octave: haven't tested it).
- To run the training scripts, you'll need:
  - `scipy` for loading the data in the `.mat` files
  - `torch` for the `Dataset` and `DataLoader` classes
  - [`wandb`](https://wandb.ai/site) for logging
  - `matplotlib` for producing the plots.
  - [`addict`](https://github.com/mewwts/addict) for cool `Dict` objects.
