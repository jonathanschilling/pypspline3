"""
3-D spline in float64 precision
"""

import numpy as _np
import warnings as _warnings

import fpspline

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
        self.bcval1min = _np.zeros( (n3, n2,) )
        self.bcval1max = _np.zeros( (n3, n2,) )

        self.bcval2min = _np.zeros( (n3, n1,) )
        self.bcval2max = _np.zeros( (n3, n1,) )

        self.bcval3min = _np.zeros( (n2, n1,) )
        self.bcval3max = _np.zeros( (n2, n1,) )

        # Compact cubic coefficient arrays
        self.__fspl = _np.zeros( (n3,n2,n1,8,) )

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

        iper=0
        if self.__ibctype1[0]==-1 or self.__ibctype1[1]==-1: iper=1
        self.__x1pkg, ifail = fpspline.genxpkg(self.__x1, iper)
        if ifail!=0:
            raise 'pspline3_r4::setup failed to compute x1pkg'

        iper=0
        if self.__ibctype2[0]==-1 or self.__ibctype2[1]==-1: iper=1
        self.__x2pkg, ifail = fpspline.genxpkg(self.__x2, iper)
        if ifail!=0:
            raise 'pspline3_r4::setup failed to compute x2pkg'

        iper=0
        if self.__ibctype3[0]==-1 or self.__ibctype3[1]==-1: iper=1
        self.__x3pkg, ifail = fpspline.genxpkg(self.__x3, iper)
        if ifail!=0:
            raise 'pspline3_r4::setup failed to compute x3pkg'

        self.__isReady = 0

        self.__fspl[:,:,:,0] = f

        self.__ilin1, self.__ilin2, self.__ilin3, ifail = \
                      fpspline.mktricub(self.__x1, self.__x2, self.__x3,
                                        self.__fspl.flat,
                                        self.__ibctype1[0], self.bcval1min.flat,
                                        self.__ibctype1[1], self.bcval1max.flat,
                                        self.__ibctype2[0], self.bcval2min.flat,
                                        self.__ibctype2[1], self.bcval2max.flat,
                                        self.__ibctype3[0], self.bcval3min.flat,
                                        self.__ibctype3[1], self.bcval3max.flat,
                                        )

        if ifail != 0 :
            raise 'pspline3_r4::setup error'

        self.__isReady = 1

    def interp_point(self, p1, p2, p3):

        """
        Point interpolation at (p1, p2, p3).
        """
        iwarn = 0
        fi,ier = fpspline.evtricub(p1, p2, p3,
                                   self.__x1, self.__x2, self.__x3,
                                   self.__ilin1, self.__ilin2, self.__ilin3,
                                   self.__fspl.flat, ICT_FVAL)
        return fi, ier, iwarn

    def interp_cloud(self, p1, p2, p3):

        """
        Cloud interpolation for all (p1[:], p2[:], p3[:]). Assume len(p1)==len(p2)==len(p3).
        """

        fi,iwarn,ier = fpspline.vectricub(ICT_FVAL, p1, p2, p3,
                                          self.__x1pkg,
                                          self.__x2pkg,
                                          self.__x3pkg,
                                          self.__fspl.flat)
        return fi, ier, iwarn

    def interp_array(self, p1, p2, p3):

        """
        Array interpolation for all (p1[i1], p2[i2], p3[i3]), i{1,2,3}=0:len( p{1,2,3} )
        """
        fi, iwarn,ier = fpspline.gridtricub(p1, p2, p3, \
                                            self.__x1pkg, \
                                            self.__x2pkg, \
                                            self.__x3pkg, \
                                            self.__fspl.flat)

        return _np.resize(fi, (len(p3), len(p2), len(p1))), ier, iwarn

    def interp(self, p1, p2, p3, meth='cloud'):

        """
        Interpolatate onto (p1, p2, p3), the coordinate-triplet which can either be a single point
        (point interpolation), 3 arrays of identical length (cloud interpolation), or 3 arrays
        of possibly different lengths (array interpolation).

        The returned value is a single float for point interpolation, it is a rank-1 array of
        length len(p1)=len(p2)=len(p3) for cloud interpolation, or a rank-3 array of shape
        (len(p3), len(p2), len(p1)) for array interpolation.

        Use meth='array' to enforce array interpolation when p1, p2 and p3 happen to have
        the same length. With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline3_r4::interp: spline coefficients were not set up!'

        if type(p1)!=type(p2) or type(p1)!=type(p3) or type(p2)!=type(p3):
            raise "pspline3_r4::interp: types (p1, p2, p3) don't match"

        if type(p1)==_np.float64:
            fi, ier, iwarn = self.interp_point(p1, p2, p3)
        else:
            if len(p1)==len(p2)==len(p3) and meth=='cloud':
                fi, ier, iwarn = self.interp_cloud(p1, p2, p3)
            else:
                fi, ier, iwarn = self.interp_array(p1, p2, p3)

        if ier:
            raise "pspline3_r4::interp error ier=%d"%ier
        if iwarn:
            _warnings.warn('pspline3_r4::interp abscissae are out of bound!')

        return fi

    def derivative_point(self, i1, i2, i3, p1, p2, p3):

        """
        Compute a single point derivative d^i1 d^i2 d^i3 f/dx1^i1 dx2^i2 dx3^i3 at (p1, p2, p3).
        Must have i{1,2,3}>=0 and i1 + i2 + i3 <=2.
        """

        iwarn = 0
        fi,ier = fpspline.evtricub(p1, p2, p3,
                                   self.__x1, self.__x2, self.__x3,
                                   self.__ilin1, self.__ilin2, self.__ilin3,
                                   self.__fspl.flat, ICT_MAP[(i1,i2,i3)])
        return fi, ier, iwarn

    def derivative_cloud(self, i1, i2, i3, p1, p2, p3):

        """
        Compute the derivative d^i1 d^i2 d^i3 f/dx1^i1 dx2^i2 dx3^i3 for a cloud (p1, p2, p3). Must have
        i{1,2,3}>=0 and i1 + i2 + i3 <=2.
        """

        fi,iwarn,ier = fpspline.vectricub(ICT_MAP[(i1,i2,i3)], p1, p2, p3,
                                          self.__x1pkg,
                                          self.__x2pkg,
                                          self.__x3pkg,
                                          self.__fspl.flat)
        return fi, ier, iwarn

    def derivative_array(self, i1, i2, i3, p1, p2, p3):

        """
        Compute the derivative d^i1 d^i2 d^i3 f/dx1^i1 dx2^i2 dx3^i3 for a grid-array (p1, p2, p3). Must have
        i{1,2,3}>=0 and i1 + i2 + i3 <=2.
        """

        xx1, xx2, xx3 = griddata(p1, p2, p3)
        fi,iwarn,ier = self.derivative_cloud(i1, i2, i3, xx1.flat, xx2.flat, xx3.flat)
        return _np.resize(fi, (len(p3), len(p2), len(p1))), ier, iwarn

    def derivative(self, i1, i2, i3, p1, p2, p3, meth='cloud'):

        """
        Compute the derivative d^i1 d^i2 d^i3 f/dx1^i1 dx2^i2 dx3^i3 at (p1, p2, p3). Must have
        i{1,2,3}>=0 and i1 + i2 + i3 <=2. See interp method for a list of possible (p1, p2, p3) shapes.
        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline3_r4::derivative: spline coefficients were not set up!'

        if type(p1)!=type(p2) or type(p1)!=type(p3) or type(p2)!=type(p3):
            raise "pspline3_r4::derivative: types (p1, p2, p3) don't match"

        if type(p1)==_np.float64:
            fi, ier, iwarn = self.derivative_point(i1,i2,i3, p1,p2,p3)
        else:
            if len(p1)==len(p2)==len(p3) and meth=='cloud':
                fi, ier, iwarn = self.derivative_cloud(i1,i2,i3, p1,p2,p3)
            else:
                fi, ier, iwarn = self.derivative_array(i1,i2,i3, p1,p2,p3)

        if ier:
            raise "pspline3_r4::derivative error"
        if iwarn:
            _warnings.warn('pspline3_r4::derivative abscissae are out of bound!')

        return fi

    def gradient_point(self, p1, p2, p3):

        """
        Return (df/dz, df/dy, df/dx) at point (p1, p2, p3).
        """

        iwarn = 0
        f1,ier1 = fpspline.evtricub(p1, p2, p3,
                                    self.__x1, self.__x2, self.__x3,
                                    self.__ilin1, self.__ilin2, self.__ilin3,
                                    self.__fspl.flat, ICT_F1)
        f2,ier2 = fpspline.evtricub(p1, p2, p3,
                                    self.__x1, self.__x2, self.__x3,
                                    self.__ilin1, self.__ilin2, self.__ilin3,
                                    self.__fspl.flat, ICT_F2)
        f3,ier3 = fpspline.evtricub(p1, p2, p3,
                                    self.__x1, self.__x2, self.__x3,
                                    self.__ilin1, self.__ilin2, self.__ilin3,
                                    self.__fspl.flat, ICT_F3)
        return f1, f2, f3, ier1+ier2+ier3, iwarn

    def gradient_cloud(self, p1, p2, p3):

        """
        Return (df/dz, df/dy, df/dx) for cloud (p1, p2, p3).
        """

        f1,iwarn1,ier1 = fpspline.vectricub(ICT_F1, p1, p2, p3,
                                            self.__x1pkg,
                                            self.__x2pkg,
                                            self.__x3pkg,
                                            self.__fspl.flat)
        f2,iwarn2,ier2 = fpspline.vectricub(ICT_F2, p1, p2, p3,
                                            self.__x1pkg,
                                            self.__x2pkg,
                                            self.__x3pkg,
                                            self.__fspl.flat)
        f3,iwarn3,ier3 = fpspline.vectricub(ICT_F3, p1, p2, p3,
                                            self.__x1pkg,
                                            self.__x2pkg,
                                            self.__x3pkg,
                                            self.__fspl.flat)
        return f1, f2, f3, ier1+ier2+ier3, iwarn1+iwarn2+iwarn3

    def gradient_array(self, p1, p2, p3):

        """
        Return (df/dz, df/dy, df/dx) for grid-array (p1, p2, p3).
        """

        xx1, xx2, xx3 = griddata(p1, p2, p3)
        f1, f2, f3, iwarn,ier = self.gradient_cloud(xx1.flat, xx2.flat, xx3.flat)
        n1, n2, n3 = len(p1), len(p2), len(p3)
        return _np.resize(f1, (n3,n2,n1)), \
               _np.resize(f2, (n3,n2,n1)), \
               _np.resize(f3, (n3,n2,n1)), \
               ier, iwarn

    def gradient(self, p1, p2, p3, meth='cloud'):

        """
        Return (df/dz, df/dy, df/dx) at point (p1, p2, p3).See interp method for a list of possible (p1, p2, p3) shapes.

        With error checks.
        """

        if self.__isReady != 1:
            raise 'pspline3_r4::gradient: spline coefficients were not set up!'

        if type(p1)!=type(p2) or type(p1)!=type(p3) or type(p2)!=type(p3):
            raise "pspline3_r4::gradient: types (p1, p2, p3) don't match"

        if type(p1)==_np.float64:
            fi, ier, iwarn = self.gradient_point(p1, p2, p3)
        else:
            if len(p1)==len(p2)==len(p3) and meth=='cloud':
                fi, ier, iwarn = self.gradient_cloud(p1, p2, p3)
            else:
                fi, ier, iwarn = self.gradient_array(p1, p2, p3)

        if ier:
            raise "pspline3_r4::gradient error"
        if iwarn:
            _warnings.warn('pspline3_r4::gradient abscissae are out of bound!')

        return fi
