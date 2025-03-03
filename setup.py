# setup.py
from setuptools import setup, Extension
import numpy

trading_core_module = Extension(
    'trading_core',
    sources=['trading_core.c'],
    include_dirs=[numpy.get_include()],
)

setup(
    name="trading_core",
    version="0.1",
    description="C extension for trading bot core functions",
    ext_modules=[trading_core_module],
    install_requires=['numpy'],
)