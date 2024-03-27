"""
Created by Elias Obreque
Date: 08-03-2024
email: els.obrq@gmail.com
"""

__all__ = ['OptimizeResult']
__docformat__ = "restructuredtext en"

import warnings
import numpy as np


# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


def _indenter(s, n=0):
    split = s.split("\n")
    indent = " "*n
    return ("\n" + indent).join(split)


def _float_formatter_10(x):
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)


def _dict_formatter(d, n=0, mplus=1, sorter=None):
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s


class OptimizeResult(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin',
                      'converged', 'flag', 'function_calls', 'iterations',
                      'root']
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown solver options: %s" % msg, None, 4)