# plat (v): plan out or make a map of

Utilities for exploring generative latent spaces.

Currently aiming to support techniques as described in the
[Sampling Generative Networks](http://arxiv.org/abs/1609.04468) paper.


## Installation

```bash
pip install plat
```

## Examples

plat can sample from a growing list of models in its model zoo. Each model type
will almost certainly have separate dependencies. For example, if
[discgen](https://github.com/dribnet/discgen) is installed then the
following would generate a random sampling of the model:
```
plat sample \
  --model celeba_64.discgen
```

It's also possible to run plat on new types of models by specifing he interface
class directly. Here's an example of how to use `plat sample` to generate a (random)
MINE grid from an iGAN model:

```bash
PYTHONPATH=. plat sample \
  --model-interface plat.interface.igan.IganModel \
  --model-file models/shoes_64.dcgan_theano \
  --uniform \
  --rows 7 --cols 13 --tight --mine --spacing 3 \
  --image-size 64 \
  --seed 1
```