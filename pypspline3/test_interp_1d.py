#!/usr/local/env python

# $Id: test_interp1_r4.py,v 1.1 2004/03/30 16:33:20 pletzer Exp $

import numpy as _np
from pypspline3.pspline_1d import pspline

import pytest

eps = _np.finfo(float).eps

def pointVal(spl, x1, ff):
    error = 0
    m1 = _np.shape(ff)[0]
    mtot = len(ff)
    for i1 in range(m1):
        fi = spl.interp(x1[i1])
        error += (fi - ff[i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def arrayVal(spl, x1, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(x1, meth='array')
    error = _np.sqrt(_np.sum( (ff - fi)**2 )/float(mtot))
    return error

def pointDx(spl, x1, ff):
    error = 0
    m1 = _np.shape(ff)[0]
    mtot = len(ff)
    for i1 in range(m1):
        fi = spl.derivative(1, x1[i1])
        error += (fi - ff[i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def arrayDx(spl, x1, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1, x1, meth='array')
    error = _np.sqrt(_np.sum( (ff - fi)**2 )/float(mtot))
    return error

def pointDxx(spl, x1, ff):
    error = 0
    m1 = _np.shape(ff)[0]
    mtot = len(ff)
    for i1 in range(m1):
        fi = spl.derivative(2, x1[i1])
        error += (fi - ff[i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def arrayDxx(spl, x1, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2, x1, meth='array')
    error = _np.sqrt(_np.sum( (ff - fi)**2 )/float(mtot))
    return error

##################################################################

# original grid
n1, n2 = 11, 21
x1 = _np.linspace(0., 1., n1)
spl = pspline(x1)
ff = x1**3
spl.setup(ff)

# new grid
x1 = _np.linspace(0., 1., n1-1)
ff = x1**3
fx = 3*x1**2
fxx= 3*2*x1

def test_pointVal():
    error = pointVal(spl, x1, ff)
    assert error < eps
    print('error = %g'%error)

def test_arrayVal():
    error = arrayVal(spl, x1, ff)
    print('error = %g'%error)
    assert error < eps

def test_pointDx():
    error = pointDx(spl, x1, fx)
    print('error = %g'%error)
    assert error < 10*eps

def test_arrayDx():
    error = arrayDx(spl, x1, fx)
    print('error = %g'%error)
    assert error < 10*eps

def test_pointDxx():
    error = pointDxx(spl, x1, fxx)
    print('error = %g'%error)
    assert error < 100*eps

def test_arrayDxx():
    error = arrayDxx(spl, x1, fxx)
    print('error = %g'%error)
    assert error < 100*eps

if __name__ == "__main__":

    print('..testing interpolation....................')
    test_pointVal()
    test_arrayVal()

    print('..testing 1st order derivatives............')
    test_pointDx()
    test_arrayDx()

    print('..testing 2nd order derivatives............')
    test_pointDxx()
    test_arrayDxx()
