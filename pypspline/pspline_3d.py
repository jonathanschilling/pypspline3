#!/usr/bin/env python

# $Id: pspline3_r4.py,v 1.2 2004/03/30 16:41:00 pletzer Exp $

"""
3-D spline in float64 precision
"""

import numpy as _np

import pspline_wrapped as fpspline

# to get the function value
ICT_FVAL = _np.array( [1,0,0,0,0,0,0,0,0,0], dtype=_np.int32 )
# 1st derivatives
ICT_F1   = _np.array( [0,1,0,0,0,0,0,0,0,0], dtype=_np.int32 )
ICT_F2   = _np.array( [0,0,1,0,0,0,0,0,0,0], dtype=_np.int32 )
ICT_F3   = _np.array( [0,0,0,1,0,0,0,0,0,0], dtype=_np.int32 )
ICT_GRAD = _np.array( [0,1,1,1,0,0,0,0,0,0], dtype=_np.int32 )
# generic derivatives
ICT_MAP = {
    (0,0,0): _np.array( [1,0,0,0,0,0,0,0,0,0], dtype=_np.int32 ),
    (1,0,0): _np.array( [0,1,0,0,0,0,0,0,0,0], dtype=_np.int32 ),
    (0,1,0): _np.array( [0,0,1,0,0,0,0,0,0,0], dtype=_np.int32 ),
    (0,0,1): _np.array( [0,0,0,1,0,0,0,0,0,0], dtype=_np.int32 ),
    (2,0,0): _np.array( [0,0,0,0,1,0,0,0,0,0], dtype=_np.int32 ),
    (0,2,0): _np.array( [0,0,0,0,0,1,0,0,0,0], dtype=_np.int32 ),
    (0,0,2): _np.array( [0,0,0,0,0,0,1,0,0,0], dtype=_np.int32 ),
    (1,1,0): _np.array( [0,0,0,0,0,0,0,1,0,0], dtype=_np.int32 ),
    (1,0,1): _np.array( [0,0,0,0,0,0,0,0,1,0], dtype=_np.int32 ),
    (0,1,1): _np.array( [0,0,0,0,0,0,0,0,0,1], dtype=_np.int32 ),
}

def griddata(x1, x2, x3):

    " Given grid vectors, return grid data "

    n1, n2, n3 = len(x1), len(x2), len(x3)

    xx1 = _np.multiply.outer( _np.ones( (n3,n2) ), x1 )
    xx2 = _np.multiply.outer( _np.ones( (n3,) ),
                             _np.multiply.outer( x2, _np.ones( (n1,)) ) )
    xx3 = _np.multiply.outer( x3, _np.ones( (n2,n1) ) )

    return xx1, xx2, xx3

###############################################################################

class pspline:

    def __init__(self, x1, x2, x3,
                 bcs1=None, bcs2=None, bcs3=None):

        """
        Constructor.

        x1, x2, x3: original grid arrays

        bcs1, bcs2, bcs3: boundary conditions. Use bcs{1,2,3}=1 to apply
        periodic boundary conditions (bcs{1,2,3} defaults to None for
        not-a-knot boundary conditions, this should be fine in most cases.

        More general boundary conditions can be applied by setting

        bcs{1,2,3}=(bmin, bmax)

        where bmin/bmax can take values from 0 to 7, as described in
        http://w3.pppl.gov/NTCC/PSPLINE/.

        The boundary conditions (if inhomogeneous) must then be applied
        by setting the class members

        self.bcval{1,2,3}min and/or self.bcval{1,2,3}max

        explicitly *prior* to calling self.setup(f).

        1 -- match slope
        2 -- match 2nd derivative
        3 -- boundary condition is slope=0
        4 -- boundary condition is d2f/dx2=0
        5 -- match 1st derivative to 1st divided difference
        6 -- match 2nd derivative to 2nd divided difference
        7 -- match 3rd derivative to 3rd divided difference

        For example, if one wishes to apply df/dx = a(x2,x3) on the left and

        d^2f/dx^2 = b(x1,x2)

        on the right of x1, use

        bcs1=(1, 2)

        and set both

        self.bcval1min = a

        and

        self.bcval1max = b

        where shape(a) == shape(b) == (n3,n2) and n{2,3} = len(x{2,3}).

        """

        self.__x1 = x1
        self.__x2 = x2
        self.__x3 = x3
        self.__n1 = len(x1)
        self.__n2 = len(x2)
        self.__n3 = len(x3)

        n3, n2, n1 = self.__n3, self.__n2, self.__n1

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

        if bcs3:
            if bcs3==1:
                # periodic
                self.__ibctype3 = (-1, -1)
            else:
                # general
                self.__ibctype3 = ( bcs3[0], bcs3[1] )
        else:
            # not-a-knot BCs
            self.__ibctype3 = (0,0)

        # BC values (see above)
        self.bcval1min = _np.zeros( (n3, n2,), order='F' )
        self.bcval1max = _np.zeros( (n3, n2,), order='F' )

        self.bcval2min = _np.zeros( (n3, n1,), order='F' )
        self.bcval2max = _np.zeros( (n3, n1,), order='F' )

        self.bcval3min = _np.zeros( (n2, n1,), order='F' )
        self.bcval3max = _np.zeros( (n2, n1,), order='F' )

        # Compact cubic coefficient arrays
        self.__fspl = _np.zeros( (8,n1,n2,n3), order='F' )

        # storage
        self.__x1pkg = None
        self.__x2pkg = None
        self.__x3pkg = None

        # check flags
        self.__isReady = 0

        # will turn out to be 1 for nearly uniform mesh
        self.__ilin1 = 0
        self.__ilin2 = 0
        self.__ilin3 = 0

    def setup(self, f):

        """
        Set up (compute) cubic spline coefficients.
        See __init__ for comment about boundary conditions.
        """

        if _np.shape(f) != (self.__n3, self.__n2, self.__n1):
            raise 'pspline3_r4::setup shape error. Got shape(f)=%s should be %s' % \
                  ( str(_np.shape(f)), str((self.__n3, self.__n2, self.__n1)) )

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
            raise 'pspline3_r4::setup failed to compute x1pkg'

        iper=0
        if self.__ibctype2[0]==-1 or self.__ibctype2[1]==-1:
            iper=1

        self.__x2pkg = _np.zeros([self.__n2, 4], order='F')

        ifail = 0
        fpspline.genxpkg(self.__n2, self.__x2, self.__x2pkg, iper,
                         imsg, itol, ztol, ialg, ifail)
        if ifail!=0:
            raise 'pspline3_r4::setup failed to compute x2pkg'

        iper=0
        if self.__ibctype3[0]==-1 or self.__ibctype3[1]==-1:
            iper=1

        self.__x3pkg = _np.zeros([self.__n3, 4], order='F')

        ifail = 0
        fpspline.genxpkg(self.__n3, self.__x3, self.__x3pkg, iper,
                         imsg, itol, ztol, ialg, ifail)
        if ifail!=0:
            raise 'pspline3_r4::setup failed to compute x3pkg'

        self.__isReady = 0

        self.__fspl[0,:,:,:] = f.T

        fpspline.mktricub(self.__x1, self.__n1,
                          self.__x2, self.__n2,
                          self.__x3, self.__n3,
                          self.__fspl, self.__n1, self.__n2,
                          self.__ibctype1[0], self.bcval1min,
                          self.__ibctype1[1], self.bcval1max, self.__n2,
                          self.__ibctype2[0], self.bcval2min,
                          self.__ibctype2[1], self.bcval2max, self.__n1,
                          self.__ibctype3[0], self.bcval3min,
                          self.__ibctype3[1], self.bcval3max, self.__n1,
                          self.__ilin1, self.__ilin2, self.__ilin3,
                          ifail)

        if ifail != 0 :
            raise 'pspline3_r4::setup error'

        self.__isReady = 1

    def interp_point(self, p1, p2, p3):

        """
        Point interpolation at (p1, p2, p3).
        """
        ier = 0
        iwarn = 0

        fi = _np.zeros(1)

        fpspline.evtricub(p1, p2, p3,
                          self.__x1, self.__n1,
                          self.__x2, self.__n2,
                          self.__x3, self.__n3,
                          self.__ilin1, self.__ilin2, self.__ilin3,
                          self.__fspl, self.__n1, self.__n2,
                          ICT_FVAL, fi, ier)
        return fi[0], ier, iwarn






###############################################################################
if __name__ == '__main__':

    import sys, time

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
