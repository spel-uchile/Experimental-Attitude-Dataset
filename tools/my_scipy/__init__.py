"""
Created by Elias Obreque
Date: 08-03-2024
email: els.obrq@gmail.com

"""


# start delvewheel patch
def _delvewheel_patch_1_5_0():
    import ctypes
    import os
    import platform
    import sys
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scipy.libs'))
    is_conda_cpython = platform.python_implementation() == 'CPython' and (hasattr(ctypes.pythonapi, 'Anaconda_GetVersion') or 'packaged by conda-forge' in sys.version)
    if sys.version_info[:2] >= (3, 8) and not is_conda_cpython or sys.version_info[:2] >= (3, 10):
        if os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)
    else:
        load_order_filepath = os.path.join(libs_dir, '.load-order-scipy-1.11.2')
        if os.path.isfile(load_order_filepath):
            with open(os.path.join(libs_dir, '.load-order-scipy-1.11.2')) as file:
                load_order = file.read().split()
            for lib in load_order:
                lib_path = os.path.join(os.path.join(libs_dir, lib))
                if os.path.isfile(lib_path) and not ctypes.windll.kernel32.LoadLibraryExW(ctypes.c_wchar_p(lib_path), None, 0x00000008):
                    raise OSError('Error loading {}; {}'.format(lib, ctypes.FormatError()))


_delvewheel_patch_1_5_0()
del _delvewheel_patch_1_5_0
# end delvewheel patch

from numpy import show_config as show_numpy_config
if show_numpy_config is None:
    raise ImportError(
        "Cannot import SciPy when running from NumPy source directory.")
from numpy import __version__ as __numpy_version__

# Import numpy symbols to scipy name space (DEPRECATED)
from ._lib.deprecation import _deprecated
import numpy as np
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.{0} instead')

# deprecate callable objects from numpy, skipping classes and modules
import types as _types  # noqa: E402
for _key in np.__all__:
    if _key.startswith('_'):
        continue
    _fun = getattr(np, _key)
    if isinstance(_fun, _types.ModuleType):
        continue
    if callable(_fun) and not isinstance(_fun, type):
        _fun = _deprecated(_msg.format(_key))(_fun)
    globals()[_key] = _fun
del np, _types

from numpy.random import rand, randn
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.random.{0} instead')
rand = _deprecated(_msg.format('rand'))(rand)
randn = _deprecated(_msg.format('randn'))(randn)

# fft is especially problematic, so was removed in SciPy 1.6.0
from numpy.fft import ifft
ifft = _deprecated('scipy.ifft is deprecated and will be removed in SciPy '
                   '2.0.0, use scipy.fft.ifft instead')(ifft)

from numpy.lib import scimath  # noqa: E402
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.lib.scimath.{0} instead')
for _key in scimath.__all__:
    _fun = getattr(scimath, _key)
    if callable(_fun):
        _fun = _deprecated(_msg.format(_key))(_fun)
    globals()[_key] = _fun
del scimath
del _msg, _fun, _key, _deprecated

# We first need to detect if we're being called as part of the SciPy
# setup procedure itself in a reliable manner.
__SCIPY_SETUP__ = True

import sys
sys.stderr.write('Running from SciPy source directory.\n')
del sys
