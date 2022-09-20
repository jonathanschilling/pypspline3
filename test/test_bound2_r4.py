#!/usr/local/env python

# $Id$

import Numeric as N
from pypspline.pspline2_r4 import pspline, griddata
EPS = 1.e-6

def linspace(xmin, xmax, nx):
    return N.arange(xmin, xmax+EPS, (xmax-xmin)/float(nx-1), typecode=N.Float32)

def periodic1():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=None)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)   
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def periodic2():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=None, bcs2=1)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def allPeriodic():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=1)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def slope1():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=(1,1), bcs2=None)
    spl.bcval1min = N.zeros( (n2,), N.Float32 )
    spl.bcval1max = N.zeros( (n2,), N.Float32 )
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def slope2():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=None, bcs2=(1,1))
    spl.bcval2min = N.cos(xx1[ 0,:])*N.cos(xx2[ 0,:])
    spl.bcval2max = N.cos(xx1[-1,:])*N.cos(xx2[-1,:])
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def secondDer1():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=(2,2), bcs2=1)
    spl.bcval1min = N.zeros( (n2,), N.Float32 )
    spl.bcval1max = N.zeros( (n2,), N.Float32 )
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))

def secondDer2():
    n1, n2 = 11, 21
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)
    spl = pspline(x1, x2, bcs1=1, bcs2=(2,2))
    spl.bcval2min = - N.cos(xx1[ 0,:])*N.sin(xx2[ 0,:])
    spl.bcval2max = - N.cos(xx1[-1,:])*N.sin(xx2[-1,:])
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = N.cos(xx1)*N.sin(xx2)    
    ffi, ier, iwarn = spl.interp_array(x1, x2)
    return N.sqrt(N.sum(N.sum( (ff-ffi)**2 ))/float(len(ff)))
    

##################################################################

if __name__=='__main__':

    import sys

    cum_error = 0
    error = periodic1()
    cum_error += error
    print 'error = %g'%error
    error = periodic2()
    cum_error += error
    print 'error = %g'%error
    error = allPeriodic()
    cum_error += error
    print 'error = %g'%error
    error = slope1()
    cum_error += error
    print 'error = %g'%error
    error = slope2()
    cum_error += error
    print 'error = %g'%error
    error = secondDer1()
    cum_error += error
    print 'error = %g'%error
    error = secondDer2()
    cum_error += error
    print 'error = %g'%error

    print 'cumulated error = %g in %s' % (cum_error, sys.argv[0])
    if cum_error > 0.007:
       print 'TEST %s FAILED' %  sys.argv[0]
