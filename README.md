# plat (v): plan out or make a map of

Utilities for exploring generative latent spaces.

Currently aiming to support techniques as described in the
[Sampling Generative Networks](http://arxiv.org/abs/1609.04468) paper.


## Installation

```bash
pip install plat
```

## Example

Here's one example of how to use plat-sample to generate a (random)
MINE grid from an iGAN model:

```bash
THEANORC=theanorc.mine PYTHONPATH=.:../iGAN \
  python ./plat-sample.py \
  --interface plat.interface.igan.IganModel \
  --uniform \
  --rows 7 --cols 13 --tight --splash --spacing 3 \
  --image-size 64 \
  --seed 1 \
  --model ../iGAN/models/shoes_64.dcgan_theano \
  --save-path "example_shoes_mine_7x13_sp3_seed01_01.png"
```