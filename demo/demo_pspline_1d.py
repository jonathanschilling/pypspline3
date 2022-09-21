#!/usr/bin/env python

"""
1-D spline in float64 precision
"""

import sys, time

import numpy as np
import matplotlib.pyplot as plt

from pypspline3.pspline_1d import pspline

eps = 1.e-6

n1 = 11
bcs1 = (0,0)
x1min, x1max = 0., 1.
x1 = np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))

tic =  time.time()
#########
f = x1**3
#########
toc = time.time()
print("function evaluations: time->%10.1f secs" % (toc-tic))

tic = time.time()
spl = pspline(x1)
# may set BCs if not-a-knot
spl.setup(f)
toc = time.time()
print("init/setup: %d original grid nodes time->%10.1f secs" % \
        (n1, toc-tic))

# save/load is not considered here

# new mesh
n1 = 2*n1-1
x2 = np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
fexact = x2**3

# point interpolation

nint = n1

#    all_fi = np.zeros(n1)

error = 0
all_fi = np.zeros(n1)
tic = time.time()
for i1 in range(n1):
    fi, ier, iwarn = spl.interp_point(x2[i1])
    error += (fi - fexact[i1])**2
    all_fi[i1] = fi
toc = time.time()
error /= nint
error = np.sqrt(error)
print("interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))

plt.figure()
plt.plot(x1, f, "o-", label="orig")
plt.plot(x2, all_fi, ".--", label="interp")
plt.legend(loc="upper left")
plt.show()

# array interpolation

tic = time.time()
fi, ier, iwarn = spl.interp_array(x2)
toc = time.time()
error = np.sum(np.sum(np.sum((fi-fexact)**2)))/nint
print("interp_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))

## df/dx

fexact = 3*x2**2

# point df/dx

tic = time.time()
error = 0
for i1 in range(n1):
    fi, ier, iwarn = spl.derivative_point(1, x2[i1])
    error += (fi - fexact[i1])**2
toc = time.time()
error /= nint
error = np.sqrt(error)
print("derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))

# array df/dx

tic = time.time()
fi, ier, iwarn = spl.derivative_array(1, x2)
toc = time.time()
error = np.sum(np.sum(np.sum((fi-fexact)**2)))/nint
print("derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))

## d^2f/dx^2

fexact = 3*2*x2

# point d^2f/dx^2

tic = time.time()
error = 0
for i1 in range(n1):
    fi, ier, iwarn = spl.derivative_point(2, x2[i1])
    error += (fi - fexact[i1])**2
toc = time.time()
error /= nint
error = np.sqrt(error)
print("derivative_point d^2f/dx^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))

# array d^2f/dx^2

tic = time.time()
fi, ier, iwarn = spl.derivative_array(2, x2)
toc = time.time()
error = np.sum(np.sum(np.sum((fi-fexact)**2)))/nint
print("derivative_array d^2f/dx^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
        (nint, error, ier, iwarn, toc-tic))
