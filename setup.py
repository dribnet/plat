from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
    'scipy',
    'Pillow'
]

setup(name='plat',
      version='0.1.0',
      description='Utilities for working with generative latent spaces',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/plat',
      download_url='https://github.com/dribnet/plat/tarball/0.1.0',
      license='MIT',
      install_requires=install_requires,
      packages=find_packages())
