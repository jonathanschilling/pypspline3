#!/usr/local/env python

# $Id$

import Numeric as N
from pspline3_r4 import pspline, griddata
EPS = 1.e-6

def linspace(xmin, xmax, nx):
    return N.arange(xmin, xmax+EPS, (xmax-xmin)/float(nx-1), typecode=N.Float32)

def periodic1():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=None, bcs3=None)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def periodic2():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=1, bcs3=None)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def periodic3():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=None, bcs3=1)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def allPeriodic():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=1, bcs3=1)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope1():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=(1,1), bcs2=None, bcs3=None)
    spl.bcval1min = N.zeros( (n3, n2), N.Float32 )
    spl.bcval1max = N.zeros( (n3, n2), N.Float32 )
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope2():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=(1,1), bcs3=None)
    spl.bcval2min = N.cos(xx1[:, 0,:])*N.cos(xx2[:, 0,:])*N.cos(2*xx3[:, 0,:])
    spl.bcval2max = N.cos(xx1[:,-1,:])*N.cos(xx2[:,-1,:])*N.cos(2*xx3[:,-1,:])
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def slope3():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=None, bcs2=None, bcs3=(1,1))
    spl.bcval3min = N.zeros( (n2, n1), N.Float32 )
    spl.bcval3max = N.zeros( (n2, n1), N.Float32 )
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def secondDer1():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=(2,2), bcs2=1, bcs3=1)
    spl.bcval1min = N.zeros( (n3, n2), N.Float32 )
    spl.bcval1max = N.zeros( (n3, n2), N.Float32 )
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

def secondDer2():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=(2,2), bcs3=1)
    spl.bcval2min = - N.cos(xx1[:, 0,:])*N.sin(xx2[:, 0,:])*N.cos(2*xx3[:, 0,:])
    spl.bcval2max = - N.cos(xx1[:,-1,:])*N.sin(xx2[:,-1,:])*N.cos(2*xx3[:,-1,:])
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))
    

def secondDer3():
    n1, n2, n3 = 11, 21, 31
    x1 = linspace(0., 2*N.pi, n1)
    x2 = linspace(0., 2*N.pi, n2)
    x3 = linspace(0., 2*N.pi, n3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)
    spl = pspline(x1, x2, x3, bcs1=1, bcs2=1, bcs3=(2,2))
    spl.bcval3min = -4 * N.cos(xx1[ 0,:,:])*N.sin(xx2[ 0,:,:])*N.cos(2*xx3[ 0,:,:])
    spl.bcval3max = -4 * N.cos(xx1[-1,:,:])*N.sin(xx2[-1,:,:])*N.cos(2*xx3[-1,:,:])
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    x2 = linspace(0., 2*N.pi, n2-1)
    x3 = linspace(0., 2*N.pi, n3-1)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = N.cos(xx1)*N.sin(xx2)*N.cos(2*xx3)    
    ffi, ier, iwarn = spl.interp_array(x1, x2, x3)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))

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
    error = periodic3()
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
    error = slope3()
    cum_error += error
    print 'error = %g'%error
    error = secondDer1()
    cum_error += error
    print 'error = %g'%error
    error = secondDer2()
    cum_error += error
    print 'error = %g'%error
    error = secondDer3()
    cum_error += error
    print 'error = %g'%error

    print 'cumulated error = %g in %s' % (cum_error, sys.argv[0])
    if cum_error > 0.30:
       print 'TEST %s FAILED' %  sys.argv[0]
