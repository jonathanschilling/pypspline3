#!/usr/bin/env python

# $Id$

"""
2-D spline in real*4 precision
"""

import Numeric as N
import MLab
import fpspline
import warnings, types, time

# to get the function value
ICT_FVAL = N.array( [1,0,0,0,0,0] )
# 1st derivatives
ICT_F1   = N.array( [0,1,0,0,0,0] )
ICT_F2   = N.array( [0,0,1,0,0,0] )
ICT_GRAD = N.array( [0,1,1,0,0,0] )
# generic derivatives
ICT_MAP = {
    (0,0): N.array( [1,0,0,0,0,0] ),
    (1,0): N.array( [0,1,0,0,0,0] ),
    (0,1): N.array( [0,0,1,0,0,0] ),
    (2,0): N.array( [0,0,0,1,0,0] ),
    (0,2): N.array( [0,0,0,0,1,0] ),
    (1,1): N.array( [0,0,0,0,0,1] ),
    }


def griddata(x1, x2):
    
    " Given grid vectors, return grid data "
    
    n1, n2 = len(x1), len(x2)

    xx1 = N.multiply.outer( N.ones( (n2,), typecode=N.Float32 ), x1 )
    xx2 = N.multiply.outer( x2, N.ones( (n1,), typecode=N.Float32 ) )

    return xx1, xx2

###############################################################################
    

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
        self.bcval1min = N.zeros( (n2,), N.Float32 )
        self.bcval1max = N.zeros( (n2,), N.Float32 )

        self.bcval2min = N.zeros( (n1,), N.Float32 )
        self.bcval2max = N.zeros( (n1,), N.Float32 )

        # Compact cubic coefficient arrays
        self.__fspl = N.zeros( (n2,n1,4,), N.Float32 )

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
        See self.__init__.doc for comment about boundary conditions. 
        """

        if N.shape(f) != (self.__n2, self.__n1):
            raise 'pspline2_r4::setup shape error. Got shape(f)=%s should be %s' % \
                  ( str(N.shape(f)), str((self.__n2, self.__n1)) )

        # default values for genxpg
        imsg=0
        itol=0        # range tolerance option
        ztol=5.e-7    # range tolerance, if itol is set
        ialg=-3       # algorithm selection code
        
        iper=0        
        if self.__ibctype1[0]==-1 or self.__ibctype1[1]==-1: iper=1
        self.__x1pkg, ifail = fpspline.genxpkg(self.__x1, iper)
        if ifail!=0:
            raise 'pspline2_r4::setup failed to compute x1pkg'
        
        iper=0
        if self.__ibctype2[0]==-1 or self.__ibctype2[1]==-1: iper=1
        self.__x2pkg, ifail = fpspline.genxpkg(self.__x2, iper)
        if ifail!=0:
            raise 'pspline2_r4::setup failed to compute x2pkg'
        
        self.__isReady = 0

        self.__fspl[:,:,0] = f

        self.__ilin1, self.__ilin2, ifail = \
                      fpspline.mkbicub(self.__x1, self.__x2, \
                                       self.__fspl.flat, \
                                       self.__ibctype1[0], self.bcval1min.flat, \
                                       self.__ibctype1[1], self.bcval1max.flat, \
                                       self.__ibctype2[0], self.bcval2min.flat, \
                                       self.__ibctype2[1], self.bcval2max.flat, \
                                       )
        
        if ifail != 0 :
            raise 'pspline2_r4::setup error'

        self.__isReady = 1


    def interp_point(self, p1, p2):

        """
        Point interpolation at (p1, p2).
        """

        iwarn = 0
        fi,ier = fpspline.evbicub(p1, p2, \
                                    self.__x1, self.__x2, \
                                    self.__ilin1, self.__ilin2, \
                                    self.__fspl.flat, ICT_FVAL)
        return fi, ier, iwarn

    def interp_cloud(self, p1, p2):

        """
        Cloud interpolation for all (p1[:], p2[:]). Assume len(p1)==len(p2).
        """

        fi,iwarn,ier = fpspline.vecbicub(ICT_FVAL, p1, p2, \
                                         self.__x1pkg, \
                                         self.__x2pkg, \
                                         self.__fspl.flat)
        return fi, ier, iwarn   

    def interp_array(self, p1, p2):

        """
        Array interpolation for all (p1[i1], p2[i2]), i{1,2}=0:len( p{1,2} )
        """

        fi, iwarn,ier = fpspline.gridbicub(p1, p2, \
                                         self.__x1pkg, \
                                         self.__x2pkg, \
                                         self.__fspl.flat)

        return N.resize(fi, (len(p2), len(p1))), ier, iwarn
        

    def interp(self, p1, p2, meth = 'cloud'):

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

        if type(p1)!=type(p2):
            raise "pspline2_r4::interp: types (p1, p2) don't match"

        if type(p1)==types.FloatType:
            fi, ier, iwarn = self.interp_point(p1, p2)
        else:
            if len(p1)==len(p2):
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

        iwarn = 0
        fi,ier = fpspline.evbicub(p1, p2, \
                                    self.__x1, self.__x2, \
                                    self.__ilin1, self.__ilin2, \
                                    self.__fspl.flat, ICT_MAP[(i1,i2)])
        return fi, ier, iwarn

    def derivative_cloud(self, i1, i2, p1, p2):

        """
        Compute the derivative d^i1 d^i2 f/dx1^i1 dx2^i2 for a cloud (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2. 
        """

        fi,iwarn,ier = fpspline.vecbicub(ICT_MAP[(i1,i2)], p1, p2, \
                                         self.__x1pkg, \
                                         self.__x2pkg, \
                                         self.__fspl.flat)
        return fi, ier, iwarn
    
    def derivative_array(self, i1, i2, p1, p2):

        """
        Compute the derivative d^i1 d^i2 f/dx1^i1 dx2^i2 for a grid-array (p1, p2). Must have
        i{1,2}>=0 and i1 + i2 <=2. 
        """

        xx1, xx2 = griddata(p1, p2)
        fi,iwarn,ier = self.derivative_cloud(i1, i2, xx1.flat, xx2.flat)
        return N.resize(fi, (len(p2), len(p1))), ier, iwarn

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

        if type(p1)==types.FloatType:
            fi, ier, iwarn = self.derivative_point(i1,i2, p1,p2)
        else:
            if len(p1)==len(p2):
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
        Return (df/dz, df/dy, df/dx) at point (p1, p2).
        """

        iwarn = 0
        f1,ier1 = fpspline.evbicub(p1, p2, \
                                    self.__x1, self.__x2, \
                                    self.__ilin1, self.__ilin2, \
                                    self.__fspl.flat, ICT_F1)
        f2,ier2 = fpspline.evbicub(p1, p2, \
                                    self.__x1, self.__x2, \
                                    self.__ilin1, self.__ilin2, \
                                    self.__fspl.flat, ICT_F2)
        return f1, f2, ier1+ier2, iwarn

    def gradient_cloud(self, p1, p2):

        """
        Return (df/dz, df/dy, df/dx) for cloud (p1, p2).
        """

        f1,iwarn1,ier1 = fpspline.vecbicub(ICT_F1, p1, p2, \
                                         self.__x1pkg, \
                                         self.__x2pkg, \
                                         self.__fspl.flat)
        f2,iwarn2,ier2 = fpspline.vecbicub(ICT_F2, p1, p2, \
                                         self.__x1pkg, \
                                         self.__x2pkg, \
                                         self.__fspl.flat)
        return f1, f2, ier1+ier2, iwarn1+iwarn2
    
    def gradient_array(self, p1, p2):

        """
        Return (df/dz, df/dy, df/dx) for grid-array (p1, p2).
        """

        xx1, xx2 = griddata(p1, p2)
        f1, f2, iwarn,ier = self.gradient_cloud(xx1.flat, xx2.flat)
        n1, n2 = len(p1), len(p2)
        return N.resize(f1, (n2,n1)), \
               N.resize(f2, (n2,n1)), \
               ier, iwarn

    def gradient(self, p1, p2, meth='cloud'):
    
        """
        """

        if self.__isReady != 1:
            raise 'pspline2_r4::gradient: spline coefficients were not set up!'

        if type(p1)!=type(p2):
            raise "pspline2_r4::gradient: types (p1, p2) don't match"

        if type(p1)==types.FloatType:
            fi, ier, iwarn = self.gradient_point(p1, p2)
        else:
            if len(p1)==len(p2):
                fi, ier, iwarn = self.gradient_cloud(p1, p2)
            else:
                fi, ier, iwarn = self.gradient_array(p1, p2)        

        if ier:
            raise "pspline2_r4::gradient error"
        if iwarn:
            warnings.warn('pspline2_r4::gradient abscissae are out of bound!')
    
        return fi


    def save(self, filename):

        """
        Save state in NetCDF file 
        """

        from Scientific.IO.NetCDF import NetCDFFile

        ncf = NetCDFFile(filename, mode='w')

        ncf.title = "pspline2_r4.save(..) file created on %s" % time.asctime()

        ncf.createDimension('_1', 1)
        ncf.createDimension('_2', 2)
        ncf.createDimension('n1', self.__n1)
        ncf.createDimension('n2', self.__n2)
        isHermite = ncf.createVariable('isHermite', N.Int, ('_1',)) 
        isReady = ncf.createVariable('isReady', N.Int, ('_1',))
        ibctype1 = ncf.createVariable('ibctype1', N.Int, ('_2',))
        ibctype2 = ncf.createVariable('ibctype2', N.Int, ('_2',))
        x1 = ncf.createVariable('x1', N.Float32, ('n1',))
        x2 = ncf.createVariable('x2', N.Float32, ('n2',))
        bcval1min = ncf.createVariable('bcval1min',  N.Float32, ('n2',))
        bcval1max = ncf.createVariable('bcval1max',  N.Float32, ('n2',))
        bcval2min = ncf.createVariable('bcval2min',  N.Float32, ('n1',))
        bcval2max = ncf.createVariable('bcval2max',  N.Float32, ('n1',))
        f = ncf.createVariable('f',  N.Float32, ('n2', 'n1'))
        isHermite.assignValue(0)
        isReady.assignValue(self.__isReady)
        ibctype1.assignValue(self.__ibctype1)
        ibctype2.assignValue(self.__ibctype2)
        x1.assignValue(self.__x1)
        x2.assignValue(self.__x2)
        bcval1min.assignValue(self.bcval1min)
        bcval1max.assignValue(self.bcval1max)
        bcval2min.assignValue(self.bcval2min)
        bcval2max.assignValue(self.bcval2max)
        f.assignValue(self.__fspl[:,:,0])
        ncf.close()

    def load(self, filename):

        """
        Save state in NetCDF file 
        """

        from Scientific.IO.NetCDF import NetCDFFile

        ncf = NetCDFFile(filename, mode='r')
        
        if ncf.variables['isHermite'][:][0] !=0:
            raise 'pspline2_r4::load incompatible interpolation method'

        self.__ibctype1 = tuple(ncf.variables['ibctype1'][:])
        self.__ibctype2 = tuple(ncf.variables['ibctype2'][:])
        self.__x1 = ncf.variables['x1'][:]
        self.__x2 = ncf.variables['x2'][:]
        self.bcval1min = ncf.variables['bcval1min'][:]
        self.bcval1max = ncf.variables['bcval1max'][:]
        self.bcval2min = ncf.variables['bcval2min'][:]
        self.bcval2max = ncf.variables['bcval2max'][:]
        f = ncf.variables['f'][:]
        ncf.close()

        self.setup(f)
        

###############################################################################
if __name__ == '__main__':

    import sys

    eps = 1.e-6

    n1, n2 = 11, 21
    bcs1 = (0,0)
    bcs2 = (0,0)   # not-a-knot
    x1min, x1max = 0., 1.
    x2min, x2max = 0., 1.
    x1 = N.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1),
                  typecode=N.Float32)
    x2 = N.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1),
                  typecode=N.Float32)
    tic =  time.time()
    xx1, xx2 = griddata(x1, x2)
    toc =  time.time()
    print "griddata            : time->%10.1f secs" % (toc-tic)
    tic = time.time()
    #########################
    f = xx1**3 + 2*xx1*xx2**3
    #########################
    toc = time.time()
    print "function evaluations: time->%10.1f secs" % (toc-tic)

    tic = time.time()
    spl = pspline(x1, x2)
    # may set BCs if not-a-knot 
    spl.setup(f.astype(N.Float32))
    toc = time.time()
    print "init/setup: %d original grid nodes time->%10.1f secs" % (n1*n2, toc-tic)

    # save/load
    tic = time.time()
    spl.save('spl.nc')
    spl.load('spl.nc')
    toc = time.time()
    print "save/load: time->%10.1f secs" % (toc-tic)

    # new mesh
    n1, n2 = 2*n2-1, 2*n2 # 3*n2-1, 3*n2
    x1 = N.arange(x1min, x1max+eps, (x1max-x1min)/float(n1-1))
    x2 = N.arange(x2min, x2max+eps, (x2max-x2min)/float(n2-1))
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
    error = N.sqrt(error)
    print "interp_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array interpolation

    tic = time.time()
    fi, ier, iwarn = spl.interp_array(x1, x2)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "interp_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # cloud interpolation

    n = n1*n2
    xc1 = N.arange(x1min, x1max+eps, (x1max-x1min)/float(n-1))
    xc2 = N.arange(x2min, x2max+eps, (x2max-x2min)/float(n-1))
    fcexact = xc1**3 + 2*xc1*xc2**3
    
    tic = time.time()
    fi, ier, iwarn = spl.interp_cloud(xc1, xc2)
    toc = time.time()
    error = N.sum((fi-fcexact)**2)/nint
    print "interp_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)
    
    
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
    error = N.sqrt(error)
    print "derivative_point df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array df/dx

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(1, 0, x1, x2)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "derivative_array df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # cloud df/dx

    fcexact = 3*xc1**2 + 2*xc2**3
    
    tic = time.time()
    fi, ier, iwarn = spl.derivative_cloud(1, 0, xc1, xc2)
    toc = time.time()
    error = N.sum((fi-fcexact)**2)/nint
    print "derivative_cloud df/dx: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)
    
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
    error = N.sqrt(error)
    print "derivative_point d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array d^2f/dy^2

    tic = time.time()
    fi, ier, iwarn = spl.derivative_array(0,2, x1, x2)
    toc = time.time()
    error = N.sum(N.sum(N.sum((fi-fexact)**2)))/nint
    print "derivative_array d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # cloud d^2f/dy^2

    fcexact = 12*xc1*xc2
    
    tic = time.time()
    fi, ier, iwarn = spl.derivative_cloud(0,2, xc1, xc2)
    toc = time.time()
    error = N.sum((fi-fcexact)**2)/nint
    print "derivative_cloud d^2f/dy^2: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)
    

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
    error = N.sqrt(error)
    print "gradient_point: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # array

    tic = time.time()
    f1,f2, ier, iwarn = spl.gradient_array(x1, x2)
    toc = time.time()
    error = N.sum(N.sum(N.sum(((f1-f1exact)**2 + (f2-f2exact)**2)/2.)))/nint
    print "gradient_array: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)

    # cloud

    f1cexact = 3*xc1**2 + 2*    xc2**3
    f2cexact =            6*xc1*xc2**2
    
    tic = time.time()
    f1,f2, ier, iwarn = spl.gradient_cloud(xc1, xc2)
    toc = time.time()
    error = N.sum(((f1-f1cexact)**2 + (f2-f2cexact)**2/2.))/nint
    print "gradient_cloud: %d evaluations (error=%g) ier=%d iwarn=%d time->%10.1f secs" % \
          (nint, error, ier, iwarn, toc-tic)
    
