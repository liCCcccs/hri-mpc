# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="HRI_ODE",
    ext_modules=cythonize("hri_ode_c_shoenonlinear.pyx"),
)
