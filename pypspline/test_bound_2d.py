#!/usr/local/env python

# $Id: test_bound2_r4.py,v 1.1 2004/03/30 16:33:19 pletzer Exp $

import numpy as _np
from pypspline.pspline_2d import pspline, griddata

EPS = 1.e-6

def periodic1():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=None)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def periodic2():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=None, bcs2=1)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def allPeriodic():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=1)
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def slope1():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=(1,1), bcs2=None)
    spl.bcval1min = _np.zeros( (n2,) )
    spl.bcval1max = _np.zeros( (n2,) )
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def slope2():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=None, bcs2=(1,1))
    spl.bcval2min = _np.cos(xx1[ 0,:])*_np.cos(xx2[ 0,:])
    spl.bcval2max = _np.cos(xx1[-1,:])*_np.cos(xx2[-1,:])
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def secondDer1():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=(2,2), bcs2=1)
    spl.bcval1min = _np.zeros( (n2,) )
    spl.bcval1max = _np.zeros( (n2,) )
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))

def secondDer2():
    n1, n2 = 11, 21
    x1 = _np.linspace(0., 2*_np.pi, n1)
    x2 = _np.linspace(0., 2*_np.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=(2,2))
    spl.bcval2min = - _np.cos(xx1[ 0,:])*_np.sin(xx2[ 0,:])
    spl.bcval2max = - _np.cos(xx1[-1,:])*_np.sin(xx2[-1,:])
    spl.setup(ff)
    x1 = _np.linspace(0., 2*_np.pi, n1-1)
    x2 = _np.linspace(0., 2*_np.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = _np.cos(xx1)*_np.sin(xx2)
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return _np.sqrt(_np.sum(_np.sum( (ff-ffi)**2 ))/float(len(ff)))


##################################################################

if __name__=='__main__':

    import sys

    cum_error = 0
    error = periodic1()
    cum_error += error
    print('error = %g'%error)
    error = periodic2()
    cum_error += error
    print('error = %g'%error)
    error = allPeriodic()
    cum_error += error
    print('error = %g'%error)
    error = slope1()
    cum_error += error
    print('error = %g'%error)
    error = slope2()
    cum_error += error
    print('error = %g'%error)
    error = secondDer1()
    cum_error += error
    print('error = %g'%error)
    error = secondDer2()
    cum_error += error
    print('error = %g'%error)

    print('cumulated error = %g in %s' % (cum_error, sys.argv[0]))
    if cum_error > 0.007:
       print('TEST %s FAILED' %  sys.argv[0])
