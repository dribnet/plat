from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
    'scipy',
    'Pillow',
    'arghandler',
    'braceexpand',
    'fuel'
]

setup(name='plat',
      version='0.2.1',
      description='Utilities for working with generative latent spaces',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/plat',
      download_url='https://github.com/dribnet/plat/archive/0.2.1.tar.gz',
      license='MIT',
      entry_points={
          'console_scripts': ['plat = plat.bin.platcmd:main']
      },
      install_requires=install_requires,
      packages=find_packages())
