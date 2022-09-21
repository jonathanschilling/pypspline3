#!/usr/local/env python

# $Id: test_interp2_r4.py,v 1.1 2004/03/30 16:33:20 pletzer Exp $

import numpy as _np
from pypspline3.pspline_2d import pspline, griddata

import pytest

eps = _np.finfo(float).eps

def pointVal(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.interp(x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudVal(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayVal(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDx(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(1,0, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDx(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0, xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayDx(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0, x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum( (ff - fi)**2 ))/float(mtot))
    return error


def pointDy(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(0,1, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1, xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayDy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1, x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum( (ff - fi)**2 ))/float(mtot))
    return error

def pointDxx(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(2,0, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDxx(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0, xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayDxx(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0, x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum( (ff - fi)**2 ))/float(mtot))
    return error

def pointDxy(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(1,1, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDxy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1, xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayDxy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1, x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum( (ff - fi)**2 ))/float(mtot))
    return error

def pointDyy(spl, x1, x2, ff):
    error = 0
    m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(0,2, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDyy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2, xx1.flatten(), xx2.flatten())
    error = _np.sqrt(_np.sum( (ff.flatten() - fi)**2 )/float(mtot))
    return error

def arrayDyy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2, x1, x2, meth='array')
    error = _np.sqrt(_np.sum(_np.sum( (ff - fi)**2 ))/float(mtot))
    return error

##################################################################

# original grid
n1, n2 = 11, 21
x1 = _np.linspace(0., 1., n1)
x2 = _np.linspace(0., 1., n2)
spl = pspline(x1, x2)
xx1, xx2 = griddata(x1, x2)
ff = xx1**3 + 2*xx2**3 + xx1**3 * xx2**2
spl.setup(ff)

# new grid
x1 = _np.linspace(0., 1., n1-1)
x2 = _np.linspace(0., 1., n2-1)
xx1, xx2 = griddata(x1, x2)
ff = xx1**3 + 2*xx2**3 + xx1**3 * xx2**2
fx = 3*xx1**2 + 3*xx1**2 * xx2**2
fy = 6*xx2**2 + 2*xx1**3 * xx2
fxx = 6*xx1 +  6*xx1 * xx2**2
fxy = 6 * xx1**2 * xx2
fyy = 12*xx2 + 2*xx1**3


def test_pointVal():
    error = pointVal(spl, x1, x2, ff)
    print('error = %g'%error)
    assert error < 10*eps

def test_cloudVal():
    error = cloudVal(spl, xx1, xx2, ff)
    print('error = %g'%error)
    assert error < 10*eps

def test_arrayVal():
    error = arrayVal(spl, x1, x2, ff)
    print('error = %g'%error)
    assert error < 10*eps

# -------------------------------------

def test_pointDx():
    error = pointDx(spl, x1, x2, fx)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_cloudDx():
    error = cloudDx(spl, xx1, xx2, fx)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_arrayDx():
    error = arrayDx(spl, x1, x2, fx)
    print('error = %g'%error)
    assert error < 10*10*eps


def test_pointDy():
    error = pointDy(spl, x1, x2, fy)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_cloudDy():
    error = cloudDy(spl, xx1, xx2, fy)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_arrayDy():
    error = arrayDy(spl, x1, x2, fy)
    print('error = %g'%error)
    assert error < 10*10*eps

# -------------------------------------

def test_pointDxx():
    error = pointDxx(spl, x1, x2, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_cloudDxx():
    error = cloudDxx(spl, xx1, xx2, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_arrayDxx():
    error = arrayDxx(spl, x1, x2, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps


def test_pointDxy():
    error = pointDxy(spl, x1, x2, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_cloudDxy():
    error = cloudDxy(spl, xx1, xx2, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_arrayDxy():
    error = arrayDxy(spl, x1, x2, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps


def test_pointDyy():
    error = pointDyy(spl, x1, x2, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_cloudDyy():
    error = cloudDyy(spl, xx1, xx2, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

def test_arrayDyy():
    error = arrayDyy(spl, x1, x2, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*eps

if __name__=='__main__':

    print('..testing interpolation....................')

    test_pointVal()
    test_cloudVal()
    test_arrayVal()

    print('..testing 1st order derivatives............')

    test_pointDx()
    test_cloudDx()
    test_arrayDx()

    test_pointDy()
    test_cloudDy()
    test_arrayDy()


    print('..testing 2nd order derivatives............')

    test_pointDxx()
    test_cloudDxx()
    test_arrayDxx()

    test_pointDxy()
    test_cloudDxy()
    test_arrayDxy()

    test_pointDyy()
    test_cloudDyy()
    test_arrayDyy()
