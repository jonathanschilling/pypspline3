#!/usr/local/env python

# $Id$

import Numeric as N
from pspline2_r8 import pspline, griddata
EPS = 1.e-10

def linspace(xmin, xmax, nx):
    return N.arange(xmin, xmax+EPS, (xmax-xmin)/float(nx-1))

def pointVal(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.interp(x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudVal(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayVal(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDx(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(1,0, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDx(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0, xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayDx(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0, x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum( (ff - fi)**2 ))/float(mtot))
    return error


def pointDy(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(0,1, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1, xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayDy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1, x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum( (ff - fi)**2 ))/float(mtot))
    return error

def pointDxx(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(2,0, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDxx(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0, xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxx(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0, x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum( (ff - fi)**2 ))/float(mtot))
    return error

def pointDxy(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(1,1, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDxy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1, xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1, x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum( (ff - fi)**2 ))/float(mtot))
    return error
    
def pointDyy(spl, x1, x2, ff):
    error = 0
    m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i2 in range(m2):
        for i1 in range(m1):
            fi = spl.derivative(0,2, x1[i1], x2[i2])
            error += (fi - ff[i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDyy(spl, xx1, xx2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2, xx1.flat, xx2.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDyy(spl, x1, x2, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2, x1, x2, meth='array')
    error = N.sqrt(N.sum(N.sum( (ff - fi)**2 ))/float(mtot))
    return error


    
##################################################################

if __name__=='__main__':

    import sys

    # original grid
    n1, n2 = 11, 21
    x1 = linspace(0., 1., n1)
    x2 = linspace(0., 1., n2)
    spl = pspline(x1, x2)
    xx1, xx2 = griddata(x1, x2)
    ff = xx1**3 + 2*xx2**3 + xx1**3 * xx2**2
    spl.setup(ff.astype(N.Float64))

    # new grid
    x1 = linspace(0., 1., n1-1)
    x2 = linspace(0., 1., n2-1)
    xx1, xx2 = griddata(x1, x2)
    ff = xx1**3 + 2*xx2**3 + xx1**3 * xx2**2
    fx = 3*xx1**2 + 3*xx1**2 * xx2**2
    fy = 6*xx2**2 + 2*xx1**3 * xx2
    fxx = 6*xx1 +  6*xx1 * xx2**2
    fxy = 6 * xx1**2 * xx2
    fyy = 12*xx2 + 2*xx1**3
    

    cum_error = 0
    
    print '..testing interpolation....................'

    error = pointVal(spl, x1, x2, ff)
    cum_error += error
    print 'error = %g'%error
    error = cloudVal(spl, xx1, xx2, ff)
    cum_error += error
    print 'error = %g'%error
    error = arrayVal(spl, x1, x2, ff)
    cum_error += error
    print 'error = %g'%error

    print '..testing 1st order derivatives............'

    error = pointDx(spl, x1, x2, fx)
    cum_error += error
    print 'error = %g'%error
    error = cloudDx(spl, xx1, xx2, fx)
    cum_error += error
    print 'error = %g'%error
    error = arrayDx(spl, x1, x2, fx)
    cum_error += error
    print 'error = %g'%error
    
    error = pointDy(spl, x1, x2, fy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDy(spl, xx1, xx2, fy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDy(spl, x1, x2, fy)
    cum_error += error
    print 'error = %g'%error

    print '..testing 2nd order derivatives............'
    
    error = pointDxx(spl, x1, x2, fxx)
    cum_error += error
    print 'error = %g'%error
    error = cloudDxx(spl, xx1, xx2, fxx)
    cum_error += error
    print 'error = %g'%error
    error = arrayDxx(spl, x1, x2, fxx)
    cum_error += error
    print 'error = %g'%error
    
    error = pointDxy(spl, x1, x2, fxy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDxy(spl, xx1, xx2, fxy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDxy(spl, x1, x2, fxy)
    cum_error += error
    print 'error = %g'%error

    error = pointDyy(spl, x1, x2, fyy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDyy(spl, xx1, xx2, fyy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDyy(spl, x1, x2, fyy)
    cum_error += error
    print 'error = %g'%error

    print 'cumulated error = %g in %s' % (cum_error, sys.argv[0])
    if cum_error > 0.30:
       print 'TEST %s FAILED' %  sys.argv[0]
