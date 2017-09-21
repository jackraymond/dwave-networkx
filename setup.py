from setuptools import setup, find_packages

from dwave_networkx import __version__

setup(
    name='dwave_networkx',
    version=__version__,
    packages=find_packages(),
    install_requires=['networkx>=2.0', 'decorator>=4.1.0']
)
