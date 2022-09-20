#!/usr/bin/env python

"""
2-D spline in float64 precision
"""

import numpy as _np

import pspline_wrapped as fpspline

# to get the function value
ICT_FVAL = _np.array([1,0,0,0,0,0], dtype=_np.int32)
# 1st derivatives
ICT_F1   = _np.array([0,1,0,0,0,0], dtype=_np.int32)
ICT_F2   = _np.array([0,0,1,0,0,0], dtype=_np.int32)
ICT_GRAD = _np.array([0,1,1,0,0,0], dtype=_np.int32)
# generic derivatives
ICT_MAP = {
    (0,0): _np.array([1,0,0,0,0,0], dtype=_np.int32),
    (1,0): _np.array([0,1,0,0,0,0], dtype=_np.int32),
    (0,1): _np.array([0,0,1,0,0,0], dtype=_np.int32),
    (2,0): _np.array([0,0,0,1,0,0], dtype=_np.int32),
    (0,2): _np.array([0,0,0,0,1,0], dtype=_np.int32),
    (1,1): _np.array([0,0,0,0,0,1], dtype=_np.int32),
}

def griddata(x1, x2):

    " Given grid vectors, return grid data "

    n1, n2 = len(x1), len(x2)

    xx1 = _np.multiply.outer( _np.ones( (n2,) ), x1 )
    xx2 = _np.multiply.outer( x2, _np.ones( (n1,) ) )

    return xx1, xx2

class pspline:

    def __init__(self, x1, x2,
                 bcs1=None, bcs2=None):

        """
        Constructor.

        x1, x2: original grid arrays

        bcs1, bcs2: boundary conditions. Use bcs{1,2}=1 to apply
        periodic boundary conditions (bcs{1,2} defaults to None for
        not-a-knot boundary conditions, this should be fine in most cases.

        More general boundary conditions can be applied by setting

        bcs{1,2}=(bmin, bmax)

        where bmin/bmax can take values from 0 to 7, as described in
        http://w3.pppl.gov/NTCC/PSPLINE/.

        The boundary conditions (if inhomogeneous) must then be applied
        by setting the class members

        self.bcval{1,2}min and/or self.bcval{1,2}max

        explicitly *prior* to calling self.setup(f).

        1 -- match slope
        2 -- match 2nd derivative
        3 -- boundary condition is slope=0
        4 -- boundary condition is d2f/dx2=0
        5 -- match 1st derivative to 1st divided difference
        6 -- match 2nd derivative to 2nd divided difference
        7 -- match 3rd derivative to 3rd divided difference

        For example, if one wishes to apply df/dx = a(x2) on the left and

        d^2f/dx^2 = b(x2)

        on the right of x1, use

        bcs1=(1, 2)

        and set both

        self.bcval1min = a

        and

        self.bcval1max = b

        where shape(a) == shape(b) == (n2,) and n2 = len(x2).

        """

        self.__x1 = x1
        self.__x2 = x2
        self.__n1 = len(x1)
        self.__n2 = len(x2)

        n2, n1 = self.__n2, self.__n1

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

        if bcs2:
            if bcs2==1:
                # periodic
                self.__ibctype2 = (-1, -1)
            else:
                # general
                self.__ibctype2 = ( bcs2[0], bcs2[1] )
        else:
            # not-a-knot BCs
            self.__ibctype2 = (0,0)


        # BC values (see above)
        self.bcval1min = _np.zeros( (n2,) )
        self.bcval1max = _np.zeros( (n2,) )

        self.bcval2min = _np.zeros( (n1,) )
        self.bcval2max = _np.zeros( (n1,) )

        # Compact cubic coefficient arrays
        self.__fspl = _np.zeros( (4,n1,n2,), order='F' )

        # storage
        self.__x1pkg = None
        self.__x2pkg = None

        # check flags
        self.__isReady = 0

        # will turn out to be 1 for nearly uniform mesh
        self.__ilin1 = 0
        self.__ilin2 = 0

    def setup(self, f):

        """
        Set up (compute) cubic spline coefficients.
        See __init__ for comment about boundary conditions.
        """

        if _np.shape(f) != (self.__n2, self.__n1):
            raise 'pspline2_r4::setup shape error. Got shape(f)=%s should be %s' % \
                  ( str(_np.shape(f)), str((self.__n2, self.__n1)) )

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
            raise 'pspline2_r4::setup failed to compute x1pkg'

        iper=0
        if self.__ibctype2[0]==-1 or self.__ibctype2[1]==-1:
            iper=1

        self.__x2pkg = _np.zeros([self.__n2, 4], order='F')

        fpspline.genxpkg(self.__n2, self.__x2, self.__x2pkg, iper,
                         imsg, itol, ztol, ialg, ifail)
        if ifail!=0:
            raise 'pspline2_r4::setup failed to compute x2pkg'

        self.__isReady = 0

        self.__fspl[0,:,:] = f.T

        fpspline.mkbicub(self.__x1, self.__n1, self.__x2, self.__n2,
                         self.__fspl, self.__n1,
                         self.__ibctype1[0], self.bcval1min,
                         self.__ibctype1[1], self.bcval1max,
                         self.__ibctype2[0], self.bcval2min,
                         self.__ibctype2[1], self.bcval2max,
                         self.__ilin1, self.__ilin2, ifail)

        if ifail != 0 :
            raise 'pspline2_r4::setup error'

        self.__isReady = 1

    def interp_point(self, p1, p2):

        """
        Point interpolation at (p1, p2).
        """

        ier = 0
        iwarn = 0

        fi = _np.zeros(1)

        fpspline.evbicub(p1, p2,
                         self.__x1, self.__n1, self.__x2, self.__n2,
                         self.__ilin1, self.__ilin2,
                         self.__fspl, self.__n1,
                         ICT_FVAL, fi, ier)

        return fi[0], ier, iwarn

    def interp_cloud(self, p1, p2):

        """
        Cloud interpolation for all (p1[:], p2[:]). Assume len(p1)==len(p2).
        """

        nEval = len(p1)
        if nEval != len(p2):
            raise RuntimeError("p1 and p2 must have equal length, but have %d and %d"%
                               (nEval, len(p2)))

        fi = _np.zeros(nEval)

        ier = 0
        iwarn = 0

        fpspline.vecbicub(ICT_FVAL, nEval, p1, p2,
                          nEval, fi,
                          self.__n1, self.__x1pkg,
                          self.__n2, self.__x2pkg,
                          self.__fspl, self.__n1, iwarn, ier)

        return fi, ier, iwarn

    def interp_array(self, p1, p2):

        """
        Array interpolation for all (p1[i1], p2[i2]), i{1,2}=0:len( p{1,2} )
        """

        n1 = len(p1)
        n2 = len(p2)

        fi = _np.zeros([n1, n2], order="F")

        ier = 0
        iwarn = 0

        fpspline.gridbicub(p1, n1, p2, n2,
                           fi, n1,
                           self.__n1, self.__x1pkg,
                           self.__n2, self.__x2pkg,
                           self.__fspl, self.__n1, iwarn, ier)

        return fi.reshape([n1, n2], order='F').T, ier, iwarn

    def interp(self, p1, p2, meth='cloud'):

        """
        Interpolatate onto (p1, p2), the coordinate-doublet which can either be a single point
        (point interpolation), 2 arrays of identical length (cloud interpolation), or 2 arrays
        of possibly different lengths (array interpolation).

        The returned value is a single float for point interpolation, it is a rank-1 array of
        length len(p1)=len(p2) for cloud interpolation, or a rank-2 array of shape
        (len(p2), len(p1)) for array interpolation.

        Use meth='array' to enforce array interpolation when p1 and p2 happen to have
        the same length. With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline2_r4::interp: spline coefficients were not set up!'

        if type(p1) != type(p2):
            raise "pspline2_r4::interp: types (p1, p2) don't match"

        if type(p1) == _np.float64:
            fi, ier, iwarn = self.interp_point(p1, p2)
        else:
            if len(p1) == len(p2) and meth=='cloud':
                fi, ier, iwarn = self.interp_cloud(p1, p2)
            else:
                fi, ier, iwarn = self.interp_array(p1, p2)

        if ier:
            raise "pspline2_r4::interp error ier=%d"%ier
        if iwarn:
            warnings.warn('pspline2_r4::interp abscissae are out of bound!')

        return fi

    def derivative_point(self, i1, i2, p1, p2):

        """
        Compute a single point derivative d^i1 d^i2 f/dx1^i1 dx2^i2 at (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2.
        """

        ier = 0
        iwarn = 0

        fi = _np.zeros(1)

        fpspline.evbicub(p1, p2,
                         self.__x1, self.__n1, self.__x2, self.__n2,
                         self.__ilin1, self.__ilin2,
                         self.__fspl, self.__n1,
                         ICT_MAP[(i1,i2)], fi, ier)

        return fi[0], ier, iwarn

    def derivative_cloud(self, i1, i2, p1, p2):

        """
        Compute the derivative d^i1 d^i2 f/dx1^i1 dx2^i2 for a cloud (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2.
        """

        nEval = len(p1)
        if nEval != len(p2):
            raise RuntimeError("p1 and p2 must have equal length, but have %d and %d"%
                               (nEval, len(p2)))

        fi = _np.zeros(nEval)

        ier = 0
        iwarn = 0

        fpspline.vecbicub(ICT_MAP[(i1,i2)], nEval, p1, p2,
                          nEval, fi,
                          self.__n1, self.__x1pkg,
                          self.__n2, self.__x2pkg,
                          self.__fspl, self.__n1, iwarn, ier)

        return fi, ier, iwarn

    def derivative_array(self, i1, i2, p1, p2):

        """
        Compute the derivative d^i1 d^i2 f/dx1^i1 dx2^i2 for a grid-array (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2.
        """

        n1 = len(p1)
        n2 = len(p2)

        xx1, xx2 = griddata(p1, p2)
        fi,iwarn,ier = self.derivative_cloud(i1, i2, xx1.flatten(), xx2.flatten())
        return fi.reshape([n1, n2], order='F').T, ier, iwarn

    def derivative(self, i1, i2, p1, p2, meth='cloud'):

        """
        Compute the derivative d^i1 d^i2 f/dx1^i1 dx2^i2 at (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2. See interp method for a list of possible (p1, p2) shapes.
        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline2_r4::derivative: spline coefficients were not set up!'

        if type(p1)!=type(p2):
            raise "pspline2_r4::derivative: types (p1, p2) don't match"

        if type(p1) == _np.float64:
            fi, ier, iwarn = self.derivative_point(i1,i2, p1,p2)
        else:
            if len(p1)==len(p2) and meth=='cloud':
                fi, ier, iwarn = self.derivative_cloud(i1,i2, p1,p2)
            else:
                fi, ier, iwarn = self.derivative_array(i1,i2, p1,p2)

        if ier:
            raise "pspline2_r4::derivative error"
        if iwarn:
            warnings.warn('pspline2_r4::derivative abscissae are out of bound!')

        return fi

    def gradient_point(self, p1, p2):

        """
        Return (df/dy, df/dx) at point (p1, p2).
        """

        iwarn = 0

        ier1 = 0
        f1 = _np.zeros(1)
        fpspline.evbicub(p1, p2,
                         self.__x1, self.__n1, self.__x2, self.__n2,
                         self.__ilin1, self.__ilin2,
                         self.__fspl, self.__n1,
                         ICT_F1, f1, ier1)

        ier2 = 0
        f2 = _np.zeros(1)
        fpspline.evbicub(p1, p2,
                         self.__x1, self.__n1, self.__x2, self.__n2,
                         self.__ilin1, self.__ilin2,
                         self.__fspl, self.__n1,
                         ICT_F2, f2, ier2)

        return f1, f2, ier1+ier2, iwarn

    def gradient_cloud(self, p1, p2):

        """
        Return (df/dy, df/dx) for cloud (p1, p2).
        """

        nEval = len(p1)
        if nEval != len(p2):
            raise RuntimeError("p1 and p2 must have equal length, but have %d and %d"%
                               (nEval, len(p2)))

        fi1 = _np.zeros(nEval)
        iwarn1 = 0
        ier1 = 0
        fpspline.vecbicub(ICT_F1, nEval, p1, p2,
                          nEval, fi1,
                          self.__n1, self.__x1pkg,
                          self.__n2, self.__x2pkg,
                          self.__fspl, self.__n1, iwarn1, ier1)

        fi2 = _np.zeros(nEval)
        iwarn2 = 0
        ier2 = 0
        fpspline.vecbicub(ICT_F2, nEval, p1, p2,
                          nEval, fi2,
                          self.__n1, self.__x1pkg,
                          self.__n2, self.__x2pkg,
                          self.__fspl, self.__n1, iwarn2, ier2)

        return fi1, fi2, ier1+ier2, iwarn1+iwarn2

    def gradient_array(self, p1, p2):

        """
        Return (df/dy, df/dx) for grid-array (p1, p2).
        """

        xx1, xx2 = griddata(p1, p2)
        f1, f2, iwarn,ier = self.gradient_cloud(xx1.flatten(), xx2.flatten())
        n1, n2 = len(p1), len(p2)
        return f1.reshape([n1, n2], order='F').T, \
               f2.reshape([n1, n2], order='F').T, \
               ier, iwarn

    def gradient(self, p1, p2, meth='cloud'):

        """
        Return (df/dy, df/dx). See interp method for a list of possible (p1, p2) shapes.

        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline2_r4::gradient: spline coefficients were not set up!'

        if type(p1)!=type(p2):
            raise "pspline2_r4::gradient: types (p1, p2) don't match"

        if type(p1)==np.float64:
            fi, ier, iwarn = self.gradient_point(p1, p2)
        else:
            if len(p1)==len(p2) and meth=='cloud':
                fi, ier, iwarn = self.gradient_cloud(p1, p2)
            else:
                fi, ier, iwarn = self.gradient_array(p1, p2)

        if ier:
            raise "pspline2_r4::gradient error"
        if iwarn:
            warnings.warn('pspline2_r4::gradient abscissae are out of bound!')

        return fi

if __name__ == '__main__':

    import sys, time

    eps = 1.e-6

    n1, n2 = 11, 21
    bcs1 = (0,0)
    bcs2 = (0,0)   # not-a-knot
    x1min, x1max = 0., 1.
    x2min, x2max = 0., 1.
    x1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
    x2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1))
    tic = time.time()
    xx1, xx2 = griddata(x1, x2)
    toc =  time.time()
    print("griddata            : time->%10.1f secs" % (toc-tic))
    tic = time.time()
    #########################
    f = xx1**3 + 2*xx1*xx2**3
    #########################
    toc = time.time()
    print("function evaluations: time->%10.1f secs" % (toc-tic))

    tic = time.time()
    spl = pspline(x1, x2)
    # may set BCs if not-a-knot
    spl.setup(f)
    toc = time.time()
    print("init/setup: %d original grid nodes time->%10.1f secs" %
          (n1*n2, toc-tic))

    # skip save/load for now

    # new mesh
    n1, n2 = 2*n2-1, 2*n2 # 3*n2-1, 3*n2
    x1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
    x2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1))
    xx1, xx2 = griddata(x1, x2)
    fexact = xx1**3 + 2*xx1*xx2**3

    # point interpolation

    nint = n1*n2

    error = 0
    tic = time.time()
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.interp_point(x1[i1], x2[i2])
            error += (fi - fexact[i2,i1])**2
    toc = time.time()
    error /= nint
    error = _np.sqrt(error)
    print("interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # cloud interpolation

    n = n1*n2
    xc1 = _np.arange(x1min, x1max+eps, (x1max-x1min)/float(n-1))
    xc2 = _np.arange(x2min, x2max+eps, (x2max-x2min)/float(n-1))
    fcexact = xc1**3 + 2*xc1*xc2**3

    tic = time.time()
    fi, ier, iwarn = spl.interp_cloud(xc1, xc2)
    toc = time.time()
    error = _np.sum((fi-fcexact)**2)/nint
    print("interp_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # array interpolation

    tic = time.time()
    fi, ier, iwarn = spl.interp_array(x1, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
    print("interp_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    ## df/dx

    fexact = 3*xx1**2 + 2*xx2**3

    # point df/dx

    tic = time.time()
    error = 0
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.derivative_point(1, 0, x1[i1], x2[i2])
            error += (fi - fexact[i2,i1])**2
    toc = time.time()
    error /= nint
    error = _np.sqrt(error)
    print("derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # array df/dx

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(1, 0, x1, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
    print("derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # cloud df/dx

    fcexact = 3*xc1**2 + 2*xc2**3

    tic = time.time()
    fi, ier, iwarn = spl.derivative_cloud(1, 0, xc1, xc2)
    toc = time.time()
    error = _np.sum((fi-fcexact)**2)/nint
    print("derivative_cloud df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    ## d^2f/dy^2

    fexact = 12*xx1*xx2

    # point d^2f/dy^2

    error = 0
    tic = time.time()
    for i2 in range(n2):
        for i1 in range(n1):
            fi, ier, iwarn = spl.derivative_point(0,2, x1[i1], x2[i2])
            error += (fi - fexact[i2,i1])**2
    toc = time.time()
    error /= nint
    error = _np.sqrt(error)
    print("derivative_point d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # array d^2f/dy^2

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(0,2, x1, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum((fi-fexact)**2)))/nint
    print("derivative_array d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # cloud d^2f/dy^2

    fcexact = 12*xc1*xc2

    tic = time.time()
    fi, ier, iwarn = spl.derivative_cloud(0,2, xc1, xc2)
    toc = time.time()
    error = _np.sum((fi-fcexact)**2)/nint
    print("derivative_cloud d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    ## gradients

    f1exact = 3*xx1**2 + 2*    xx2**3
    f2exact =            6*xx1*xx2**2

    # point

    error = 0
    tic = time.time()
    for i2 in range(n2):
        for i1 in range(n1):
            f1,f2, ier, iwarn = spl.gradient_point(x1[i1], x2[i2])
            error += ( \
                (f1 - f1exact[i2,i1])**2 +
                (f2 - f2exact[i2,i1])**2
                )/2.
    toc = time.time()
    error /= nint
    error = _np.sqrt(error)
    print("gradient_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # array

    tic = time.time()
    f1,f2, ier, iwarn = spl.gradient_array(x1, x2)
    toc = time.time()
    error = _np.sum(_np.sum(_np.sum(((f1-f1exact)**2 + (f2-f2exact)**2)/2.)))/nint
    print("gradient_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

    # cloud

    f1cexact = 3*xc1**2 + 2*    xc2**3
    f2cexact =            6*xc1*xc2**2

    tic = time.time()
    f1,f2, ier, iwarn = spl.gradient_cloud(xc1, xc2)
    toc = time.time()
    error = _np.sum(((f1-f1cexact)**2 + (f2-f2cexact)**2/2.))/nint
    print("gradient_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" %
          (nint, error, ier, iwarn, toc-tic))

