#!/usr/bin/env python

# $Id$

"""
1-D spline in real*4 precision
"""

import Numeric as N
import MLab
import fpspline
import warnings, types, time

# to get the function value
ICT_FVAL = N.array( [1,0,0] )
# 1st derivatives
ICT_F1   = N.array( [0,1] )
ICT_GRAD = N.array( [0,1] )
# generic derivatives
ICT_MAP = {
    0: N.array( [1,0,0] ),
    1: N.array( [0,1,0] ),
    2: N.array( [0,0,1] ),
    }

###############################################################################
    

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
        self.__fspl = N.zeros( (n1,2,), N.Float32 )

        # storage
        self.__x1pkg = None

        # check flags
        self.__isReady = 0

        # will turn out to be 1 for nearly uniform mesh
        self.__ilin1 = 0
        
                

    def setup(self, f):

        """
        Set up (compute) cubic spline coefficients.
        See self.__init__.doc for comment about boundary conditions. 
        """

        if N.shape(f) != (self.__n1,):
            raise 'pspline1_r4::setup shape error. Got shape(f)=%s should be %s' % \
                  ( str(N.shape(f)), str((self.__n1,)) )

        # default values for genxpg
        imsg=0
        itol=0        # range tolerance option
        ztol=5.e-7    # range tolerance, if itol is set
        ialg=-3       # algorithm selection code
        
        iper=0        
        if self.__ibctype1[0]==-1 or self.__ibctype1[1]==-1: iper=1
        self.__x1pkg, ifail = fpspline.genxpkg(self.__x1, iper)
        if ifail!=0:
            raise 'pspline1_r4::setup failed to compute x1pkg'
        
        self.__isReady = 0

        self.__fspl[:,0] = f

        self.__ilin1, ifail = \
                      fpspline.mkspline(self.__x1, \
                                       self.__fspl.flat, \
                                       self.__ibctype1[0], self.bcval1min.flat, \
                                       self.__ibctype1[1], self.bcval1max.flat, \
                                       )
        
        if ifail != 0 :
            raise 'pspline1_r4::setup error'

        self.__isReady = 1


    def interp_point(self, p1):

        """
        Point interpolation at p1.
        """

        iwarn = 0
        fi,ier = fpspline.evspline(p1, \
                                    self.__x1, \
                                    self.__ilin1, \
                                    self.__fspl.flat, ICT_FVAL)
        return fi, ier, iwarn

    def interp_cloud(self, p1):

        """
        Cloud interpolation for all p1[:].
        """

        fi,iwarn,ier = fpspline.vecspline(ICT_FVAL, p1, \
                                         self.__x1pkg, \
                                         self.__fspl.flat)
        return fi, ier, iwarn   

    def interp_array(self, p1):

        """
        Array interpolation for all p1[i1], i1=0:len( p1 ).
        In 1-D, this is the same as interp_cloud.
        """

        fi, iwarn,ier = fpspline.vecspline(p1, \
                                         self.__x1pkg, \
                                         self.__fspl.flat)

        return fi, ier, iwarn
        

    def interp(self, p1, *meth):

        """
        Interpolatate onto p1, the coordinate which can either be a single point
        (point interpolation) or an array  (cloud/array interpolation).

        The returned value is a single float for point interpolation,
        it is a rank-1 array of length len(p1) for cloud/array interpolation.

        The *meth argument is not used here.

        the same length. With checks enabled.

        
        """

        if self.__isReady != 1:
            raise 'pspline1_r4::interp: spline coefficients were not set up!'

        if type(p1)==types.FloatType:
            fi, ier, iwarn = self.interp_point(p1)
        else:
            fi, ier, iwarn = self.interp_cloud(p1, p2)        

        if ier:
            raise "pspline1_r4::interp error ier=%d"%ier
        if iwarn:
            warnings.warn('pspline1_r4::interp abscissae are out of bound!')
    
        return fi

    def derivative_point(self, i1, p1):

        """
        Compute a single point derivative d^i1 f/dx1^i1 at p1. Must have
        i1>=0 and i1<=2. 
        """

        iwarn = 0
        fi,ier = fpspline.evspline(p1, \
                                    self.__x1, \
                                    self.__ilin1, \
                                    self.__fspl.flat, ICT_MAP[i1])
        return fi, ier, iwarn

    def derivative_cloud(self, i1, p1):

        """
        Compute the derivative d^i1 f/dx1^i1 for a cloud p1. Must have
        i1>=0 and i1<=2.
        """

        fi,iwarn,ier = fpspline.vecspline(ICT_MAP[i1], p1, \
                                         self.__x1pkg, \
                                         self.__fspl.flat)
        return fi, ier, iwarn
    
    def derivative_array(self, i1, p1):

        """
        Compute the derivative d^i1 f/dx1^i1 for a grid-array p1. Must have
        i1>=0 and i1<=2. Same as derivative_cloud in 1-D.
        """

        fi,iwarn,ier = fpspline.vecspline(ICT_MAP[i1], p1, \
                                         self.__x1pkg, \
                                         self.__fspl.flat)
        return fi, ier, iwarn

    def derivative(self, i1, p1, *meth):
    
        """
        Compute the derivative d^i1 f/dx1^i1 at p1. Must have
        i1>=0 and i1<=2. See interp method for a list of possible p1 shapes.
        With checks enabled.
        """

        if self.__isReady != 1:
            raise 'pspline1_r4::derivative: spline coefficients were not set up!'

        if type(p1)==types.FloatType:
            fi, ier, iwarn = self.derivative_point(i1,p1)
        else:
            fi, ier, iwarn = self.derivative_cloud(i1,p1)

        if ier:
            raise "pspline1_r4::derivative error"
        if iwarn:
            warnings.warn('pspline1_r4::derivative abscissae are out of bound!')
    
        return fi
        



    def save(self, filename):

        """
        Save state in NetCDF file 
        """

        from Scientific.IO.NetCDF import NetCDFFile

        ncf = NetCDFFile(filename, mode='w')

        ncf.title = "pspline1_r4.save(..) file created on %s" % time.asctime()

        ncf.createDimension('_1', 1)
        ncf.createDimension('_2', 2)
        ncf.createDimension('n1', self.__n1)
        isHermite = ncf.createVariable('isHermite', N.Int, ('_1',)) 
        isReady = ncf.createVariable('isReady', N.Int, ('_1',))
        ibctype1 = ncf.createVariable('ibctype1', N.Int, ('_2',))
        x1 = ncf.createVariable('x1', N.Float32, ('n1',))
        bcval1min = ncf.createVariable('bcval1min',  N.Float32, ('_1',))
        bcval1max = ncf.createVariable('bcval1max',  N.Float32, ('_1',))
        f = ncf.createVariable('f',  N.Float32, ('n1',))
        isHermite.assignValue(0)
        isReady.assignValue(self.__isReady)
        ibctype1.assignValue(self.__ibctype1)
        x1.assignValue(self.__x1)
        bcval1min.assignValue(self.bcval1min)
        bcval1max.assignValue(self.bcval1max)
        f.assignValue(self.__fspl[:,0])
        ncf.close()

    def load(self, filename):

        """
        Save state in NetCDF file 
        """

        from Scientific.IO.NetCDF import NetCDFFile

        ncf = NetCDFFile(filename, mode='r')
        
        if ncf.variables['isHermite'][:][0] !=0:
            raise 'pspline1_r4::load incompatible interpolation method'

        self.__ibctype1 = tuple(ncf.variables['ibctype1'][:])
        self.__x1 = ncf.variables['x1'][:]
        self.bcval1min = ncf.variables['bcval1min'][:]
        self.bcval1max = ncf.variables['bcval1max'][:]
        f = ncf.variables['f'][:]
        ncf.close()

        self.setup(f)
        

###############################################################################
if __name__ == '__main__':

    import sys

    eps = 1.e-6

    n1 = 11
    bcs1 = (0,0)
    x1min, x1max = 0., 1.
    x1 = N.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1),
                  typecode=N.Float32)

    tic =  time.time()
    #########
    f = x1**3
    #########
    toc = time.time()
    print "function evaluations: time->%10.1f secs" % (toc-tic)

    tic = time.time()
    spl = pspline(x1)
    # may set BCs if not-a-knot 
    spl.setup(f.astype(N.Float32))
    toc = time.time()
    print "init/setup: %d original grid nodes time->%10.1f secs" % \
          (n1, toc-tic)

    # save/load
    tic = time.time()
    spl.save('spl.nc')
    spl.load('spl.nc')
    toc = time.time()
    print "save/load: time->%10.1f secs" % (toc-tic)

    # new mesh
    n1 = 2*n1-1
    x1 = N.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
    fexact = x1**3

    # point interpolation

    nint = n1
    
    error = 0
    tic = time.time()
    for i1 in range(n1):
        fi, ier, iwarn = spl.interp_point(x1[i1])
        error += (fi - fexact[i1])**2
    toc = time.time()
    error /= nint
    error = N.sqrt(error)
    print "interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array interpolation

    tic = time.time()
    fi, ier, iwarn = spl.interp_array(x1)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "interp_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    
    ## df/dx
    
    fexact = 3*x1**2

    # point df/dx

    tic = time.time()
    error = 0
    for i1 in range(n1):
        fi, ier, iwarn = spl.derivative_point(1, x1[i1])
        error += (fi - fexact[i1])**2
    toc = time.time()
    error /= nint
    error = N.sqrt(error)
    print "derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array df/dx

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(1, x1)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    ## d^2f/dx^2
    
    fexact = 6*x1

    # point d^2f/dx^2

    tic = time.time()
    error = 0
    for i1 in range(n1):
        fi, ier, iwarn = spl.derivative_point(2, x1[i1])
        error += (fi - fexact[i1])**2
    toc = time.time()
    error /= nint
    error = N.sqrt(error)
    print "derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array d^2f/dx^2

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(2, x1)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    
    
