#!/usr/bin/env python

"""
1-D spline in float64 precision
"""

import numpy as _np

import fpspline

# to get the function value
ICT_FVAL = _np.array( [1,0,0], dtype=_np.int32)
# 1st derivatives
ICT_F1   = _np.array( [0,1], dtype=_np.int32)
ICT_GRAD = _np.array( [0,1], dtype=_np.int32)
# generic derivatives
ICT_MAP = {
    0: _np.array( [1,0,0], dtype=_np.int32),
    1: _np.array( [0,1,0], dtype=_np.int32),
    2: _np.array( [0,0,1], dtype=_np.int32),
}

class pspline:
    def __init__(self, x1,
                 bcs1=None):

        """
        Constructor.

        x1: original grid array

        bcs1: boundary conditions. Use bcs1=1 to apply
        periodic boundary conditions (bcs1 defaults to None for
        not-a-knot boundary conditions, this should be fine in most cases.
        More general boundary conditions can be applied by setting
        bcs1=(bmin, bmax)

        where bmin/bmax can take values from 0 to 7, as described in
        http://w3.pppl.gov/NTCC/PSPLINE/.

        The boundary conditions (if inhomogeneous) must then be applied
        by setting the class members

        self.bcval1min and/or self.bcval1max
        explicitly *prior* to calling self.setup(f).

        1 -- match slope
        2 -- match 2nd derivative
        3 -- boundary condition is slope=0
        4 -- boundary condition is d2f/dx2=0
        5 -- match 1st derivative to 1st divided difference
        6 -- match 2nd derivative to 2nd divided difference
        7 -- match 3rd derivative to 3rd divided difference
        For example, if one wishes to apply df/dx = a on the left and
        d^2f/dx^2 = b
        on the right of x1, use
        bcs1=(1, 2)
        and set both
        self.bcval1min = a
        and
        self.bcval1max = b
        The returned value is a spline object.
        """

        self.__x1 = x1
        self.__n1 = len(x1)

        n1 = self.__n1

        # BC types
        # use these to set the boundary conditions,
        # e.g. ibctype1=(1,0) sets the 1st derivative to the left
        # but uses not-a-knot Bcs on the right. The value of the
        # derivative would eb set via bcval1min

        if bcs1:
            if bcs1==1:
                # periodic
                self.__ibctype1 = (-1, -1)
            else:
                # general
                self.__ibctype1 = ( bcs1[0], bcs1[1] )
        else:
            # not-a-knot BCs
            self.__ibctype1 = (0,0)

        # BC values (see above)
        self.bcval1min = 0
        self.bcval1max = 0

        # Compact cubic coefficient arrays
        self.__fspl = _np.zeros((2,n1), order='F')

        # storage
        self.__x1pkg = None

        # check flags
        self.__isReady = 0

        # will turn out to be 1 for nearly uniform mesh
        self.__ilin1 = 0

    def setup(self, f):

        """
        Set up (compute) cubic spline coefficients.
        See __init__ for comment about boundary conditions.
        Input is f[ix], a rank-1 array for the function values.
        """

        if _np.shape(f) != (self.__n1,):
            raise 'pspline1_r4::setup shape error. Got shape(f)=%s should be %s' % \
                  ( str(_np.shape(f)), str((self.__n1,)) )

        # default values for genxpg
        imsg=0
        itol=0        # range tolerance option
        ztol=5.e-7    # range tolerance, if itol is set
        ialg=-3       # algorithm selection code

        iper=0
        if self.__ibctype1[0]==-1 or self.__ibctype1[1]==-1:
            iper=1

        self.__x1pkg = _np.zeros([self.__n1, 4], order='F')

        ifail = 0
        fpspline.genxpkg(self.__n1, self.__x1, self.__x1pkg, iper,
                         imsg, itol, ztol, ialg, ifail)
        if ifail!=0:
            raise 'pspline1_r4::setup failed to compute x1pkg'

        self.__isReady = 0

        self.__fspl[0,:] = f

        fpspline.mkspline(self.__x1, self.__n1, self.__fspl,
                          self.__ibctype1[0], self.bcval1min,
                          self.__ibctype1[1], self.bcval1max,
                          self.__ilin1, ifail)
        if ifail != 0 :
            raise 'pspline1_r4::setup mkspline error'

        self.__isReady = 1

    def interp_point(self, p1):

        """
        Point interpolation at p1.
        """

        iwarn = 0

        fi = _np.zeros(1)
        ier = 0
        fpspline.evspline(p1,
                          self.__x1, self.__n1,
                          self.__ilin1, self.__fspl,
                          ICT_FVAL, fi, ier)

        return fi[0], ier, iwarn

    def interp_cloud(self, p1):

        """
        Cloud interpolation for all p1[:].
        In 1-D, this is the same as interp_array.
        Return the interpolated function, an error flag  (=0 if ok) and a warning flag (=0 if ok).
        """

        nEval = len(p1)

        fi = _np.zeros(nEval)

        iwarn = 0
        ier = 0
        fpspline.vecspline(ICT_FVAL,
                           nEval, p1,
                           nEval, fi,
                           self.__n1, self.__x1pkg,
                           self.__fspl, iwarn, ier)

        return fi, ier, iwarn

    def interp_array(self, p1):

        """
        Array interpolation for all p1[i1], i1=0:len( p1 ).
        In 1-D, this is the same as interp_cloud.
        Return the interpolated function, an error flag  (=0 if ok) and a warning flag (=0 if ok).
        """

        nEval = len(p1)

        fi = _np.zeros(nEval)

        iwarn = 0
        ier = 0
        fpspline.vecspline(ICT_FVAL,
                           nEval, p1,
                           nEval, fi,
                           self.__n1, self.__x1pkg,
                           self.__fspl, iwarn, ier)

        return fi, ier, iwarn

    def interp(self, p1, meth=None):

        """
        Interpolatate onto p1, the coordinate which can either be a single point
        (point interpolation) or an array  (cloud/array interpolation).
        The returned value is a single float for point interpolation,
        it is a rank-1 array of length len(p1) for cloud/array interpolation.
        The meth argument has no effect, its purpose is to provide compatibility
        with higher order spline methods.
        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline1_r4::interp: spline coefficients were not set up!'

        if type(p1) == _np.float64:
            fi, ier, iwarn = self.interp_point(p1)
        else:
            fi, ier, iwarn = self.interp_cloud(p1)

        if ier:
            raise "pspline1_r4::interp error ier=%d"%ier
        if iwarn:
            warnings.warn('pspline1_r4::interp abscissae are out of bound!')

        return fi

    def derivative_point(self, i1, p1):

        """
        Compute a single point derivative d^i1 f/dx1^i1 at p1.
        Must have i1>=0 and i1<=2.
        Return the interpolated function, an error flag  (=0 if ok) and a warning flag (=0 if ok).
        """

        ier = 0
        iwarn = 0

        fi = _np.zeros(1)
        fpspline.evspline(p1,
                          self.__x1, self.__n1,
                          self.__ilin1, self.__fspl, ICT_MAP[i1],
                          fi, ier)

        return fi[0], ier, iwarn

    def derivative_cloud(self, i1, p1):

        """
        Compute the derivative d^i1 f/dx1^i1 for a cloud p1.
        Must have i1>=0 and i1<=2.
        Return the interpolated function, an error flag  (=0 if ok) and a warning flag (=0 if ok).
        """

        nEval = len(p1)

        ier = 0
        iwarn = 0

        fi = _np.zeros(nEval)

        fpspline.vecspline(ICT_MAP[i1],
                           nEval, p1,
                           nEval, fi,
                           self.__n1, self.__x1pkg,
                           self.__fspl, iwarn, ier)

        return fi, ier, iwarn

    def derivative_array(self, i1, p1):

        """
        Compute the derivative d^i1 f/dx1^i1 for a grid-array p1. Must have
        i1>=0 and i1<=2. Same as derivative_cloud in 1-D.
        Return the interpolated function, an error flag  (=0 if ok) and a warning flag (=0 if ok).
        """

        nEval = len(p1)

        ier = 0
        iwarn = 0

        fi = _np.zeros(nEval)

        fpspline.vecspline(ICT_MAP[i1],
                           nEval, p1,
                           nEval, fi,
                           self.__n1, self.__x1pkg,
                           self.__fspl, iwarn, ier)

        return fi, ier, iwarn

    def derivative(self, i1, p1, meth=None):

        """
        Compute the derivative d^i1 f/dx1^i1 at p1. Must have
        i1>=0 and i1<=2. See interp method for a list of possible p1 shapes.

        The meth argument has no effect, its purpose is to provide compatibility
        with higher order spline methods.

        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline1_r4::derivative: spline coefficients were not set up!'

        if type(p1) == _np.float64:
            fi, ier, iwarn = self.derivative_point(i1,p1)
        else:
            fi, ier, iwarn = self.derivative_cloud(i1,p1)

        if ier:
            raise "pspline1_r4::derivative error"
        if iwarn:
            warnings.warn('pspline1_r4::derivative abscissae are out of bound!')

        return fi

if __name__ == '__main__':

    import sys, time

    eps = 1.e-6

    n1 = 11
    bcs1 = (0,0)
    x1min, x1max = 0., 1.
    x1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))

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
    x2 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
    fexact = x2**3

    # point interpolation

    nint = n1

#    all_fi = _np.zeros(n1)

    error = 0
    tic = time.time()
    for i1 in range(n1):
        fi, ier, iwarn = spl.interp_point(x2[i1])
        error += (fi - fexact[i1])**2
#        all_fi[i1] = fi
    toc = time.time()
    error /= nint
    error = _np.sqrt(error)
    print("interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic))

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x1, f, "o-")
    # plt.plot(x2, all_fi, ".--")
    # plt.show()

    # array interpolation

    tic = time.time()
    fi, ier, iwarn = spl.interp_array(x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
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
    error = _np.sqrt(error)
    print("derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic))

    # array df/dx

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(1, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
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
    error = _np.sqrt(error)
    print("derivative_point d^2f/dx^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic))

    # array d^2f/dx^2

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(2, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
    print("derivative_array d^2f/dx^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic))
