#!/usr/local/env python

# $Id: test_interp1_r4.py,v 1.1 2004/03/30 16:33:20 pletzer Exp $

import numpy as _np
from pspline_1d import pspline
EPS = 1.e-10

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

if __name__=='__main__':

    import sys

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


    cum_error = 0

    print('..testing interpolation....................')

    error = pointVal(spl, x1, ff)
    cum_error += error
    print('error = %g'%error)
    error = arrayVal(spl, x1, ff)
    cum_error += error
    print('error = %g'%error)

    print('..testing 1st order derivatives............')

    error = pointDx(spl, x1, fx)
    cum_error += error
    print('error = %g'%error)
    error = arrayDx(spl, x1, fx)
    cum_error += error
    print('error = %g'%error)

    print('..testing 2nd order derivatives............')

    error = pointDxx(spl, x1, fxx)
    cum_error += error
    print('error = %g'%error)
    error = arrayDxx(spl, x1, fxx)
    cum_error += error
    print('error = %g'%error)

    print('cumulated error = %g in %s' % (cum_error, sys.argv[0]))
    if cum_error > 0.30:
       print('TEST %s FAILED' %  sys.argv[0])
