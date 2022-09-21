#!/usr/local/env python

# $Id: test_bound3_r4.py,v 1.1 2004/03/30 16:33:20 pletzer Exp $

import numpy as _np
from pypspline3.pspline_3d import pspline, griddata

import pytest

def periodic1():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=None, bcs3=None)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def periodic2():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=1, bcs3=None)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def periodic3():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=None, bcs3=1)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def allPeriodic():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=1, bcs3=1)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope1():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=(1,1), bcs2=None, bcs3=None)
    spl.bcval1min = _np.zeros( (n3, n2), order='F' )
    spl.bcval1max = _np.zeros( (n3, n2), order='F' )
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope2():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=(1,1), bcs3=None)
    spl.bcval2min = _np.cos(xx1[:, 0,:])*_np.cos(xx2[:, 0,:])*_np.cos(2*xx3[:, 0,:])
    spl.bcval2max = _np.cos(xx1[:,-1,:])*_np.cos(xx2[:,-1,:])*_np.cos(2*xx3[:,-1,:])
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope3():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=None, bcs3=(1,1))
    spl.bcval3min = _np.zeros( (n2, n1), order='F' )
    spl.bcval3max = _np.zeros( (n2, n1), order='F' )
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def secondDer1():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=(2,2), bcs2=1, bcs3=1)
    spl.bcval1min = _np.zeros( (n3, n2), order='F' )
    spl.bcval1max = _np.zeros( (n3, n2), order='F' )
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def secondDer2():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=(2,2), bcs3=1)
    spl.bcval2min = - _np.cos(xx1[:, 0,:])*_np.sin(xx2[:, 0,:])*_np.cos(2*xx3[:, 0,:])
    spl.bcval2max = - _np.cos(xx1[:,-1,:])*_np.sin(xx2[:,-1,:])*_np.cos(2*xx3[:,-1,:])
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

def secondDer3():
    n1, n2, n3 = 11, 21, 31
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    x3 = _np.linspace(0., 2*_np.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=1, bcs3=(2,2))
    spl.bcval3min = -4 * _np.cos(xx1[ 0,:,:])*_np.sin(xx2[ 0,:,:])*_np.cos(2*xx3[ 0,:,:])
    spl.bcval3max = -4 * _np.cos(xx1[-1,:,:])*_np.sin(xx2[-1,:,:])*_np.cos(2*xx3[-1,:,:])
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    x3 = _np.linspace(0., 2*_np.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = _np.cos(xx1)*_np.sin(xx2)*_np.cos(2*xx3)
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return _np.sqrt(_np.sum(_np.sum(_np.sum( (ff-ffi)**2 )))/float(len(ff)))

##################################################################

def test_periodic1():
    error = periodic1()
    print('error = %g'%error)
    assert error < 2e-3

def test_periodic2():
    error = periodic2()
    print('error = %g'%error)
    assert error < 3e-3

def test_periodic3():
    error = periodic3()
    print('error = %g'%error)
    assert error < 3e-3

def test_allPeriodic():
    error = allPeriodic()
    print('error = %g'%error)
    assert error < 2e-3

def test_slope1():
    error = slope1()
    print('error = %g'%error)
    assert error < 2e-3

def test_slope2():
    error = slope2()
    print('error = %g'%error)
    assert error < 3e-3

def test_slope3():
    error = slope3()
    print('error = %g'%error)
    assert error < 3e-3

def test_secondDer1():
    error = secondDer1()
    print('error = %g'%error)
    assert error < 1e-2

def test_secondDer2():
    error = secondDer2()
    print('error = %g'%error)
    assert error < 2e-3

def test_secondDer3():
    error = secondDer3()
    print('error = %g'%error)
    assert error < 2e-3

if __name__=='__main__':

    test_periodic1()
    test_periodic2()
    test_periodic3()
    test_allPeriodic()

    test_slope1()
    test_slope2()
    test_slope3()

    test_secondDer1()
    test_secondDer2()
    test_secondDer3()
