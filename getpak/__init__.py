__package__ = 'getpak'
__version__ = '0.0.5'
__all__ = ['automation', 'cluster', 'commons', 'input', 'inversion_functions', 'methods', 'output', 'validation']

from getpak.input import GRS
# from getpak.output import Raster

from dask import compute
from dask import delayed

