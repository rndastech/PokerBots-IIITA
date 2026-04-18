"""
Cython build script for Hydra Engine.
Compiles hydra_engine.pyx → C shared object (.so) for maximum speed.
Run: python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="hydra_engine",
    ext_modules=cythonize(
        Extension("hydra_engine", ["hydra_engine.pyx"]),
        language_level=3,
    ),
)
