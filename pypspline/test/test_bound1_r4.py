#!/usr/local/env python

# $Id$

import Numeric as N
from pypspline.pspline1_r4 import pspline
EPS = 1.e-6

def linspace(xmin, xmax, nx):
    return N.arange(xmin, xmax+EPS, (xmax-xmin)/float(nx-1), typecode=N.Float32)

def periodic1():
    n1 = 11
    x1 = linspace(0., 2*N.pi, n1)
    ff = N.cos(x1)
    spl = pspline(x1, bcs1=1)
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    ff = N.cos(x1)   
    ffi, ier, iwarn = spl.interp_array(x1)
    return N.sqrt(N.sum(N.sum(N.sum( (ff-ffi)**2 )))/float(len(ff)))


def slope1():
    n1 = 11
    x1 = linspace(0., 2*N.pi, n1)
    ff = N.cos(x1)
    spl = pspline(x1, bcs1=(1,1))
    spl.bcval1min = 0.
    spl.bcval1max = 0.
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    ff = N.cos(x1)    
    ffi, ier, iwarn = spl.interp_array(x1)
    return N.sqrt(N.sum( (ff-ffi)**2 )/float(len(ff)))

def secondDer1():
    n1 = 11
    x1 = linspace(0., 2*N.pi, n1)
    ff = N.cos(x1)
    spl = pspline(x1, bcs1=(2,2))
    spl.bcval1min = 0.
    spl.bcval1max = 0.
    spl.setup(ff.astype(N.Float32))
    x1 = linspace(0., 2*N.pi, n1-1)
    ff = N.cos(x1)    
    ffi, ier, iwarn = spl.interp_array(x1)
    return N.sqrt(N.sum( (ff-ffi)**2 )/float(len(ff)))

##################################################################

if __name__=='__main__':

    import sys

    cum_error = 0
    error = periodic1()
    cum_error += error
    print 'error = %g'%error
    error = slope1()
    cum_error += error
    print 'error = %g'%error
    error = secondDer1()
    cum_error += error
    print 'error = %g'%error

    print 'cumulated error = %g in %s' % (cum_error, sys.argv[0])
    if cum_error > 0.002:
       print 'TEST %s FAILED' %  sys.argv[0]
