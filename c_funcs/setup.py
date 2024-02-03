# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="HRI_ODE",
    ext_modules=cythonize("hri_ode_rddq5.pyx"),
)

"""
setup(
    name="HRI_ODE",
    ext_modules=cythonize(
        "hri_ode_hddq2.pyx",
        "hri_ode_rddd2.pyx",
        "hri_ode_rddd3.pyx",
        "hri_ode_rddq4.pyx",
        "hri_ode_rddq5.pyx",
    ),
)
"""
