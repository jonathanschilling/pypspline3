#!/usr/bin/env python

"""
3-D spline in float64 precision
"""

import sys, time

import numpy as _np

from pypspline.pspline_3d import pspline, griddata

eps = 1.e-6

n1, n2, n3 = 11, 21, 31
bcs1 = (0,0)
bcs2 = (0,0)   # not-a-knot
bcs3 = (0,0)   #
x1min, x1max = 0., 1.
x2min, x2max = 0., 1.
x3min, x3max = 0., 1. # 2*_np.pi
x1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
x2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1))
x3 = _np.arange(x3min, x3max+eps, (x3max-x3min)/float(n3-1))
tic =  time.time()
xx1, xx2, xx3 = griddata(x1, x2, x3)
toc =  time.time()
print("griddata            : time->%10.1f secs" % (toc-tic))
tic = time.time()
####################################
f = xx1**3 + 2*xx2**3 + 3*xx2*xx3**2
####################################
toc = time.time()
print("function evaluations: time->%10.1f secs" % (toc-tic))

tic = time.time()
spl = pspline(x1, x2, x3)
# may set BCs if not-a-knot
spl.setup(f)
toc = time.time()
print("init/setup: %d original grid nodes time->%10.1f secs" %
        (n1*n2*n3, toc-tic))

# save/load skipped for now

# new mesh
n1, n2, n3 = 2*n2-1, 2*n2, 2*n2+1 # 3*n2-1, 3*n2, 3*n2+1
x1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
x2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1))
x3 = _np.arange(x3min, x3max+eps, (x3max-x3min)/float(n3-1))
xx1, xx2, xx3 = griddata(x1, x2, x3)
fexact = xx1**3 + 2*xx2**3 + 3*xx2*xx3**2

# point interpolation

nint = n1*n2*n3

error = 0
tic = time.time()
for i3 in range(n3):
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.interp_point(x1[i1], x2[i2], x3[i3])
            error += (fi - fexact[i3,i2,i1])**2
toc = time.time()
error /= nint
error = _np.sqrt(error)
print("interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# array interpolation

tic = time.time()
fi, ier, iwarn = spl.interp_array(x1, x2, x3)
toc = time.time()
error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
print("interp_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# cloud interpolation

n = n1*n2*n3
xc1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n-1))
xc2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n-1))
xc3 = _np.arange(x3min, x3max+eps, (x3max-x3min)/float(n-1))
fcexact = xc1**3 + 2*xc2**3 + 3*xc2*xc3**2

tic = time.time()
fi, ier, iwarn = spl.interp_cloud(xc1, xc2, xc3)
toc = time.time()
error = _np.sum((fi-fcexact)**2)/nint
print("interp_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

## df/dx

fexact = 3*xx1**2

# point df/dx

tic = time.time()
error = 0
for i3 in range(n3):
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.derivative_point(1, 0, 0, x1[i1], x2[i2], x3[i3])
            error += (fi - fexact[i3,i2,i1])**2
toc = time.time()
error /= nint
error = _np.sqrt(error)
print("derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# array df/dx

tic = time.time()
fi, ier, iwarn = spl.derivative_array(1, 0, 0, x1, x2, x3)
toc = time.time()
error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
print("derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# cloud df/dx

fcexact = 3*xc1**2

tic = time.time()
fi, ier, iwarn = spl.derivative_cloud(1, 0, 0, xc1, xc2, xc3)
toc = time.time()
error = _np.sum((fi-fcexact)**2)/nint
print("derivative_cloud df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

## d^2f/dy^2

fexact = 12*xx2

# point d^2f/dy^2

error = 0
tic = time.time()
for i3 in range(n3):
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.derivative_point(0,2,0, x1[i1], x2[i2], x3[i3])
            error += (fi - fexact[i3,i2,i1])**2
toc = time.time()
error /= nint
error = _np.sqrt(error)
print("derivative_point d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# array d^2f/dy^2

tic = time.time()
fi, ier, iwarn = spl.derivative_array(0,2,0, x1, x2, x3)
toc = time.time()
error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
print("derivative_array d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# cloud d^2f/dy^2

fcexact = 12*xc2

tic = time.time()
fi, ier, iwarn = spl.derivative_cloud(0,2,0, xc1, xc2, xc3)
toc = time.time()
error = _np.sum((fi-fcexact)**2)/nint
print("derivative_cloud d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

## d^2f/dydz

fexact = 6*xx3

# point d^2f/dydz

error = 0
tic = time.time()
for i3 in range(n3):
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.derivative_point(0,1,1, x1[i1], x2[i2], x3[i3])
            error += (fi - fexact[i3,i2,i1])**2
toc = time.time()
error /= nint
error = _np.sqrt(error)
print("derivative_point d^2f/dydz: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# array d^2f/dydz

tic = time.time()
fi, ier, iwarn = spl.derivative_array(0,1,1, x1, x2, x3)
toc = time.time()
error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
print("derivative_array d^2f/dydz: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# cloud d^2f/dydz

fcexact = 6*xc3

tic = time.time()
fi, ier, iwarn = spl.derivative_cloud(0,1,1, xc1, xc2, xc3)
toc = time.time()
error = _np.sum((fi-fcexact)**2)/nint
print("derivative_cloud d^2f/dydz: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

## gradients

f1exact = 3*xx1**2
f2exact = 6*xx2**2 + 3*xx3**2
f3exact = 6*xx2*xx3

# point

error = 0
tic = time.time()
for i3 in range(n3):
    for i2 in range(n2):
        for i1 in range(n1):
            f1,f2,f3, ier, iwarn = spl.gradient_point(x1[i1], x2[i2], x3[i3])
            error += ( \
                (f1 - f1exact[i3,i2,i1])**2 +
                (f2 - f2exact[i3,i2,i1])**2 + \
                (f3 - f3exact[i3,i2,i1])**2
                )/3
toc = time.time()
error /= nint
error = _np.sqrt(error)
print("gradient_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# array

tic = time.time()
f1,f2,f3, ier, iwarn = spl.gradient_array(x1, x2, x3)
toc = time.time()
error = _np.sum(_np.sum(_np.sum(((f1-f1exact)**2 + (f2-f2exact)**2 + (f3-f3exact)**2)/3)))/nint
print("gradient_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))

# cloud

f1cexact = 3*xc1**2
f2cexact = 6*xc2**2 + 3*xc3**2
f3cexact = 6*xc2*xc3

tic = time.time()
f1,f2,f3, ier, iwarn = spl.gradient_cloud(xc1, xc2, xc3)
toc = time.time()
error = _np.sum(((f1-f1cexact)**2 + (f2-f2cexact)**2 + (f3-f3cexact)**2)/3)/nint
print("gradient_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
        (nint, error, ier, iwarn, toc-tic))
