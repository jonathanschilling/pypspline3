#!/usr/local/env python

# $Id: test_interp3_r4.py,v 1.1 2004/03/30 16:33:20 pletzer Exp $

import numpy as _np
from pypspline3.pspline_3d import pspline, griddata

import pytest

eps = _np.finfo(float).eps

def pointVal(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.interp(x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudVal(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayVal(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDx(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,0,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDx(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,0, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDx(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,0, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,1,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,0, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,0, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,0,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,1, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,1, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDxx(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(2,0,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDxx(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0,0, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxx(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0,0, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDxy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,1,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDxy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1,0, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1,0, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDyy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,2,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDyy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2,0, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDyy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2,0, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDyz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,1,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDyz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,1, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDyz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,1, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDzz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,0,2, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDzz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,2, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDzz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,2, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDxz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = _np.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,0,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = _np.sqrt(error)
    return error

def cloudDxz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,1, xx1.flat, xx2.flat, xx3.flat)
    error = _np.sqrt(_np.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,1, x1, x2, x3, meth='array')
    error = _np.sqrt(_np.sum(_np.sum(_np.sum( (ff - fi)**2 )))/float(mtot))
    return error

##################################################################

# original grid
n1, n2, n3 = 11, 21, 31
x1 = _np.linspace(0., 1., n1)
x2 = _np.linspace(0., 1., n2)
x3 = _np.linspace(0., 1., n3)
spl = pspline(x1, x2, x3)
xx1, xx2, xx3 = griddata(x1, x2, x3)
ff = xx1**3 + 2*xx2**3 + 3*xx3**3 + xx1**3 * xx2**2 * xx3
spl.setup(ff)

# new grid
x1 = _np.linspace(0., 1., n1-1)
x2 = _np.linspace(0., 1., n2-1)
x3 = _np.linspace(0., 1., n3-1)
xx1, xx2, xx3 = griddata(x1, x2, x3)
ff = xx1**3 + 2*xx2**3 + 3*xx3**3 + xx1**3 * xx2**2 * xx3
fx = 3*xx1**2 + 3*xx1**2 * xx2**2 * xx3
fy = 6*xx2**2 + 2*xx1**3 * xx2 * xx3
fz = 9*xx3**2 +   xx1**3 * xx2**2
fxx = 6*xx1 +  6*xx1 * xx2**2 * xx3
fxy = 6 * xx1**2 * xx2 * xx3
fyy = 12*xx2 + 2*xx1**3 * xx3
fxz = 3*xx1**2 * xx2**2
fzz = 18*xx3
fyz = 2*xx1**3 * xx2


def test_pointVal():
    error = pointVal(spl, x1, x2, x3, ff)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_cloudVal():
    error = cloudVal(spl, xx1, xx2, xx3, ff)
    print('error = %g'%error)
    assert error < 10*10*eps

def test_arrayVal():
    error = arrayVal(spl, x1, x2, x3, ff)
    print('error = %g'%error)
    assert error < 10*10*eps

# -------------------------------------

def test_pointDx():
    error = pointDx(spl, x1, x2, x3, fx)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_cloudDx():
    error = cloudDx(spl, xx1, xx2, xx3, fx)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_arrayDx():
    error = arrayDx(spl, x1, x2, x3, fx)
    print('error = %g'%error)
    assert error < 10*10*10*eps


def test_pointDy():
    error = pointDy(spl, x1, x2, x3, fy)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_cloudDy():
    error = cloudDy(spl, xx1, xx2, xx3, fy)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_arrayDy():
    error = arrayDy(spl, x1, x2, x3, fy)
    print('error = %g'%error)
    assert error < 10*10*10*eps


def test_pointDz():
    error = pointDz(spl, x1, x2, x3, fz)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_cloudDz():
    error = cloudDz(spl, xx1, xx2, xx3, fz)
    print('error = %g'%error)
    assert error < 10*10*10*eps

def test_arrayDz():
    error = arrayDz(spl, x1, x2, x3, fz)
    print('error = %g'%error)
    assert error < 10*10*10*eps

# -------------------------------------

def test_pointDxx():
    error = pointDxx(spl, x1, x2, x3, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDxx():
    error = cloudDxx(spl, xx1, xx2, xx3, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDxx():
    error = arrayDxx(spl, x1, x2, x3, fxx)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps


def test_pointDxy():
    error = pointDxy(spl, x1, x2, x3, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDxy():
    error = cloudDxy(spl, xx1, xx2, xx3, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDxy():
    error = arrayDxy(spl, x1, x2, x3, fxy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps


def test_pointDxz():
    error = pointDxz(spl, x1, x2, x3, fxz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDxz():
    error = cloudDxz(spl, xx1, xx2, xx3, fxz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDxz():
    error = arrayDxz(spl, x1, x2, x3, fxz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps


def test_pointDyy():
    error = pointDyy(spl, x1, x2, x3, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDyy():
    error = cloudDyy(spl, xx1, xx2, xx3, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDyy():
    error = arrayDyy(spl, x1, x2, x3, fyy)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps


def test_pointDyz():
    error = pointDyz(spl, x1, x2, x3, fyz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDyz():
    error = cloudDyz(spl, xx1, xx2, xx3, fyz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDyz():
    error = arrayDyz(spl, x1, x2, x3, fyz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps


def test_pointDzz():
    error = pointDzz(spl, x1, x2, x3, fzz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_cloudDzz():
    error = cloudDzz(spl, xx1, xx2, xx3, fzz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

def test_arrayDzz():
    error = arrayDzz(spl, x1, x2, x3, fzz)
    print('error = %g'%error)
    assert error < 10*10*10*10*10*eps

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

    test_pointDz()
    test_cloudDz()
    test_arrayDz()

    print('..testing 2nd order derivatives............')

    test_pointDxx()
    test_cloudDxx()
    test_arrayDxx()

    test_pointDxy()
    test_cloudDxy()
    test_arrayDxy()

    test_pointDxz()
    test_cloudDxz()
    test_arrayDxz()

    test_pointDyy()
    test_cloudDyy()
    test_arrayDyy()

    test_pointDyz()
    test_cloudDyz()
    test_arrayDyz()

    test_pointDzz()
    test_cloudDzz()
    test_arrayDzz()
