import warnings
# from . import _minpack

import numpy as np
from numpy import (atleast_1d, triu, shape, transpose,
                   asarray,
                   finfo, inexact, issubdtype, dtype)
from numpy import linalg
from numpy.linalg import LinAlgError
from ._optimize import OptimizeResult, _check_unknown_options


__all__ = ['fsolve', 'leastsq']


def _check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    if (output_shape is not None) and (shape(res) != output_shape):
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = "{}: there is a mismatch between the input and output " \
                  "shape of the '{}' argument".format(checker, argname)
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += "."
            msg += f'Shape should be {output_shape} but it is {shape(res)}.'
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return shape(res), dt


def fsolve(func, x0, args=(), fprime=None, full_output=0,
           col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
           epsfcn=None, factor=100, diag=None):
    options = {'col_deriv': col_deriv,
               'xtol': xtol,
               'maxfev': maxfev,
               'band': band,
               'eps': epsfcn,
               'factor': factor,
               'diag': diag}

    res = _root_hybr(func, x0, args, jac=fprime, **options)
    if full_output:
        x = res['x']
        info = {k: res.get(k)
                    for k in ('nfev', 'njev', 'fjac', 'r', 'qtf') if k in res}
        info['fvec'] = res['fun']
        return x, info, res['status'], res['message']
    else:
        status = res['status']
        msg = res['message']
        if status == 0:
            raise TypeError(msg)
        elif status == 1:
            pass
        elif status in [2, 3, 4, 5]:
            warnings.warn(msg, RuntimeWarning)
        else:
            raise TypeError(msg)
        return res['x']


def _root_hybr(func, x0, args=(), jac=None,
               col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None,
               factor=100, diag=None, **unknown_options):
    _check_unknown_options(unknown_options)
    epsfcn = eps

    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('fsolve', 'func', func, x0, args, n, (n,))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    Dfun = jac
    if Dfun is None:
        if band is None:
            ml, mu = -10, -10
        else:
            ml, mu = band[:2]
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev,
                                 ml, mu, epsfcn, factor, diag)
    else:
        _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
        if (maxfev == 0):
            maxfev = 100 * (n + 1)
        retval = _minpack._hybrj(func, Dfun, x0, args, 1,
                                 col_deriv, xtol, maxfev, factor, diag)

    x, status = retval[0], retval[-1]

    errors = {0: "Improper input parameters were entered.",
              1: "The solution converged.",
              2: "The number of calls to function has "
                  "reached maxfev = %d." % maxfev,
              3: "xtol=%f is too small, no further improvement "
                  "in the approximate\n  solution "
                  "is possible." % xtol,
              4: "The iteration is not making good progress, as measured "
                  "by the \n  improvement from the last five "
                  "Jacobian evaluations.",
              5: "The iteration is not making good progress, "
                  "as measured by the \n  improvement from the last "
                  "ten iterations.",
              'unknown': "An error occurred."}

    info = retval[1]
    info['fun'] = info.pop('fvec')
    sol = OptimizeResult(x=x, success=(status == 1), status=status)
    sol.update(info)
    try:
        sol['message'] = errors[status]
    except KeyError:
        sol['message'] = errors['unknown']

    return sol


LEASTSQ_SUCCESS = [1, 2, 3, 4]
LEASTSQ_FAILURE = [5, 6, 7, 8]


def leastsq(func, x0, args=(), Dfun=None, full_output=False,
            col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    m = shape[0]

    if n > m:
        raise TypeError(f"Improper input: func input vector length N={n} must"
                        f" not exceed func output vector length M={m}")

    if epsfcn is None:
        epsfcn = finfo(dtype).eps

    if Dfun is None:
        if maxfev == 0:
            maxfev = 200*(n + 1)
        retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol,
                                 gtol, maxfev, epsfcn, factor, diag)
    else:
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        retval = _minpack._lmder(func, Dfun, x0, args, full_output,
                                 col_deriv, ftol, xtol, gtol, maxfev,
                                 factor, diag)

    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol, None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  "the sum of squares\n  are at most {:f} and the "
                  "relative error between two consecutive "
                  "iterates is at \n  most {:f}".format(ftol, xtol), None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible." % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError]}

    # The FORTRAN return value (possible return values are >= 0 and <= 8)
    info = retval[-1]

    if full_output:
        cov_x = None
        if info in LEASTSQ_SUCCESS:
            # This was
            # perm = take(eye(n), retval[1]['ipvt'] - 1, 0)
            # r = triu(transpose(retval[1]['fjac'])[:n, :])
            # R = dot(r, perm)
            # cov_x = inv(dot(transpose(R), R))
            # but the explicit dot product was not necessary and sometimes
            # the result was not symmetric positive definite. See gh-4555.
            perm = retval[1]['ipvt'] - 1
            n = len(perm)
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            inv_triu = linalg.get_lapack_funcs('trtri', (r,))
            try:
                # inverse of permuted matrix is a permuation of matrix inverse
                invR, trtri_info = inv_triu(r)  # default: upper, non-unit diag
                if trtri_info != 0:  # explicit comparison for readability
                    raise LinAlgError(f'trtri returned info {trtri_info}')
                invR[perm] = invR.copy()
                cov_x = invR @ invR.T
            except (LinAlgError, ValueError):
                pass
        return (retval[0], cov_x) + retval[1:-1] + (errors[info][0], info)
    else:
        if info in LEASTSQ_FAILURE:
            warnings.warn(errors[info][0], RuntimeWarning)
        elif info == 0:
            raise errors[info][1](errors[info][0])
        return retval[0], info
