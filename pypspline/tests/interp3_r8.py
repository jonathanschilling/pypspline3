#!/usr/local/env python

# $Id$

import Numeric as N
from pspline3_r8 import pspline, griddata
EPS = 1.e-10

def linspace(xmin, xmax, nx):
    return N.arange(xmin, xmax+EPS, (xmax-xmin)/float(nx-1))

def pointVal(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.interp(x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudVal(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayVal(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.interp(x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDx(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,0,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDx(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,0, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayDx(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,0, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,1,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,0, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayDy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,0, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,0,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,1, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error
    
def arrayDz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,1, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error


def pointDxx(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(2,0,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDxx(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0,0, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxx(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(2,0,0, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDxy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,1,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDxy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1,0, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,1,0, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error
    
def pointDyy(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,2,0, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDyy(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2,0, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDyy(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,2,0, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDyz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,1,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDyz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,1, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDyz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,1,1, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDzz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(0,0,2, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDzz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,2, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDzz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(0,0,2, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error

def pointDxz(spl, x1, x2, x3, ff):
    error = 0
    m3, m2, m1 = N.shape(ff)
    mtot = len(ff)
    for i3 in range(m3):
        for i2 in range(m2):
            for i1 in range(m1):
                fi = spl.derivative(1,0,1, x1[i1], x2[i2], x3[i3])
                error += (fi - ff[i3,i2,i1])**2
    error /= float(mtot)
    error = N.sqrt(error)
    return error

def cloudDxz(spl, xx1, xx2, xx3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,1, xx1.flat, xx2.flat, xx3.flat)
    error = N.sqrt(N.sum( (ff.flat - fi)**2 )/float(mtot))
    return error

def arrayDxz(spl, x1, x2, x3, ff):
    error = 0
    mtot = len(ff)
    fi = spl.derivative(1,0,1, x1, x2, x3, meth='array')
    error = N.sqrt(N.sum(N.sum(N.sum( (ff - fi)**2 )))/float(mtot))
    return error
    
##################################################################

if __name__=='__main__':

    import sys

    # original grid
    n1, n2, n3 = 6,5,4 #11, 21, 31
    x1 = linspace(0., 1., n1)
    x2 = linspace(0., 1., n2)
    x3 = linspace(0., 1., n3)
    spl = pspline(x1, x2, x3)
    xx1, xx2, xx3 = griddata(x1, x2, x3)
    ff = xx1**3 + 2*xx2**3 + 3*xx3**3 + xx1**3 * xx2**2 * xx3
    spl.setup(ff)

    # new grid
    x1 = linspace(0., 1., n1-1)
    x2 = linspace(0., 1., n2-1)
    x3 = linspace(0., 1., n3-1)
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

    cum_error = 0
        
    print '..testing interpolation....................'

    error = pointVal(spl, x1, x2, x3, ff)
    cum_error += error
    print 'error = %g'%error
    error = cloudVal(spl, xx1, xx2, xx3, ff)
    cum_error += error
    print 'error = %g'%error
    error = arrayVal(spl, x1, x2, x3, ff)
    cum_error += error
    print 'error = %g'%error

    print '..testing 1st order derivatives............'

    error = pointDx(spl, x1, x2, x3, fx)
    cum_error += error
    print 'error = %g'%error
    error = cloudDx(spl, xx1, xx2, xx3, fx)
    cum_error += error
    print 'error = %g'%error
    error = arrayDx(spl, x1, x2, x3, fx)
    cum_error += error
    print 'error = %g'%error
    
    error = pointDy(spl, x1, x2, x3, fy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDy(spl, xx1, xx2, xx3, fy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDy(spl, x1, x2, x3, fy)
    cum_error += error
    print 'error = %g'%error

    error = pointDz(spl, x1, x2, x3, fz)
    cum_error += error
    print 'error = %g'%error
    error = cloudDz(spl, xx1, xx2, xx3, fz)
    cum_error += error
    print 'error = %g'%error
    error = arrayDz(spl, x1, x2, x3, fz)
    cum_error += error
    print 'error = %g'%error
    
    print '..testing 2nd order derivatives............'
    
    error = pointDxx(spl, x1, x2, x3, fxx)
    cum_error += error
    print 'error = %g'%error
    error = cloudDxx(spl, xx1, xx2, xx3, fxx)
    cum_error += error
    print 'error = %g'%error
    error = arrayDxx(spl, x1, x2, x3, fxx)
    cum_error += error
    print 'error = %g'%error
    
    error = pointDxy(spl, x1, x2, x3, fxy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDxy(spl, xx1, xx2, xx3, fxy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDxy(spl, x1, x2, x3, fxy)
    cum_error += error
    print 'error = %g'%error

    error = pointDyy(spl, x1, x2, x3, fyy)
    cum_error += error
    print 'error = %g'%error
    error = cloudDyy(spl, xx1, xx2, xx3, fyy)
    cum_error += error
    print 'error = %g'%error
    error = arrayDyy(spl, x1, x2, x3, fyy)
    cum_error += error
    print 'error = %g'%error

    error = pointDxz(spl, x1, x2, x3, fxz)
    cum_error += error
    print 'error = %g'%error
    error = cloudDxz(spl, xx1, xx2, xx3, fxz)
    cum_error += error
    print 'error = %g'%error
    error = arrayDxz(spl, x1, x2, x3, fxz)
    cum_error += error
    print 'error = %g'%error

    error = pointDyz(spl, x1, x2, x3, fyz)
    cum_error += error
    print 'error = %g'%error
    error = cloudDyz(spl, xx1, xx2, xx3, fyz)
    cum_error += error
    print 'error = %g'%error
    error = arrayDyz(spl, x1, x2, x3, fyz)
    cum_error += error
    print 'error = %g'%error

    error = pointDzz(spl, x1, x2, x3, fzz)
    cum_error += error
    print 'error = %g'%error
    error = cloudDzz(spl, xx1, xx2, xx3, fzz)
    cum_error += error
    print 'error = %g'%error
    error = arrayDzz(spl, x1, x2, x3, fzz)
    cum_error += error
    print 'error = %g'%error

    print 'cumulated error = %g in %s' % (cum_error, sys.argv[0])
    if cum_error > 0.30:
       print 'TEST %s FAILED' %  sys.argv[0]
