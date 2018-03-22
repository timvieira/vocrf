import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(name='vocrf',
      author='Ryan Cotterell and Tim Vieira',
      description='Variable-order CRFs.',
      version='1.0',
      install_requires=[
          'lazygrad',
          'arsenal',
          'path.py',
      ],
      dependency_links=[
          'https://github.com/timvieira/lazygrad.git',
          'https://github.com/timvieira/arsenal.git',
      ],
      packages=['vocrf'],
      include_dirs=[np.get_include()],
      ext_modules = cythonize(['vocrf/**/*.pyx'])
)
