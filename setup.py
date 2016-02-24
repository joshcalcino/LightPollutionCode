from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = 'skyb',
      ext_modules = cythonize("skyb.pyx"),
      )