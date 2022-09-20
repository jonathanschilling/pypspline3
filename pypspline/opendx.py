#!/usr/bin/env python

# $Id$

"""
Save field in openDx file format. 
"""

import time, sys
import Numeric as N
from Scientific.IO.NetCDF import NetCDFFile
script_name = sys.argv[0]

def scalarField3_array_netcdf(x1, x2, x3, field, name):

    " field(x1, x2, x3) -> name.nc file "

    nc = NetCDFFile(name+'.nc','w')

    ndims = 3

    # dimensions
    n1, n2, n3 = len(x1), len(x2), len(x3)
    nc.createDimension('nx', n1)
    nc.createDimension('ny', n2)
    nc.createDimension('nz', n3)
    nc.createDimension('ndims', ndims)

    # grid
    nc_x = nc.createVariable('x', N.Float32, ('nx', 'ndims',))
    nc_y = nc.createVariable('y', N.Float32, ('ny', 'ndims',))
    nc_z = nc.createVariable('z', N.Float32, ('nz', 'ndims',))
    nc_x[:,0] = x1.astype(N.Float32); nc_x[:,1] = 0.                  ; nc_x[:,2] = 0.
    nc_y[:,0] = 0.                  ; nc_y[:,1] = x2.astype(N.Float32); nc_y[:,2] = 0.
    nc_z[:,0] = 0.                  ; nc_z[:,1] = 0.                  ; nc_z[:,2] = x3.astype(N.Float32)

    # field
    nc_field = nc.createVariable(name, N.Float32, ('nz', 'ny', 'nx',))
    nc_field.field = name
    nc_field.positions = 'z, product; y, product; x, product'
    nc_field[:,:,:] = field.astype(N.Float32)

    nc.close()

def scalarField3_array(x1, x2, x3, field, name):

    " field(x1, x2, x3) -> name.dx file "

    f = file(name+'.dx', 'w')
    print >>f, '# OpenDX file '
    funct_name = sys._getframe().f_code.co_name
    print >>f, '# created by ', script_name, '::', funct_name, ' on ' + time.asctime()
    print >>f, '# 3-d positions'
    n1, n2, n3 = len(x1), len(x2), len(x3)
    print >>f, 'object 1 class array type float rank 1 shape 3 items %d data follows' \
          % (n1*n2*n3)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                print >>f, '%g %g %g'%(x1[i1],x2[i2],x3[i3])
    print >>f, '# regular connections'
    print >>f, 'object 2 class gridconnections counts %d %d %d'%(n1,n2,n3)
    print >>f, 'object 3 class array type float rank 0 items %d data follows' \
          % (n1*n2*n3)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                print >>f, '%g' % field[i3, i2, i1]
    print >>f,"""#
object "irreg positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
#
end
"""

def scalarField2_array_netcdf(x1, x2, field, name):

    " field(x1, x2) -> name.nc file "

    nc = NetCDFFile(name+'.nc','w')

    ndims = 2

    # dimensions
    n1, n2 = len(x1), len(x2)
    nc.createDimension('nx', n1)
    nc.createDimension('ny', n2)
    nc.createDimension('ndims', ndims)

    # grid
    nc_x = nc.createVariable('x', N.Float32, ('nx', 'ndims',))
    nc_y = nc.createVariable('y', N.Float32, ('ny', 'ndims',))

    nc_x[:,0] = x1.astype(N.Float32); nc_x[:,1] = 0.
    nc_y[:,0] = 0.                  ; nc_y[:,1] = x2.astype(N.Float32)

    # field
    nc_field = nc.createVariable(name, N.Float32, ('ny', 'nx',))
    nc_field.field = name
    nc_field.positions = 'y, product; x, product'
    nc_field[:,:] = field.astype(N.Float32)

    nc.close()

def scalarField2_array(x1, x2, field, name):

    " field(x1, x2) -> name.dx file "

    f = file(name+'.dx', 'w')
    print >>f, '# OpenDX file '
    funct_name = sys._getframe().f_code.co_name
    print >>f, '# created by ', script_name, '::', funct_name, ' on ' + time.asctime()
    print >>f, '# 2-d positions'
    n1, n2 = len(x1), len(x2)
    print >>f, 'object 1 class array type float rank 1 shape 2 items %d data follows' \
          % (n1*n2)
    for i1 in range(n1):
        for i2 in range(n2):
            print >>f, '%g %g'%(x1[i1],x2[i2])
    print >>f, '# regular connections'
    print >>f, 'object 2 class gridconnections counts %d %d'%(n1,n2)
    print >>f, 'object 3 class array type float rank 0 items %d data follows' \
          % (n1*n2)
    for i1 in range(n1):
        for i2 in range(n2):
            print >>f, '%g' % field[i2, i1]
    print >>f,"""#
object "irreg positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
#
end
"""

def vectorField3_array_netcdf(x1, x2, x3, field1, field2, field3, name):

    " (field1, field2, field3)(x1, x2, x3) -> name.nc file "

    nc = NetCDFFile(name+'.nc','w')

    ndims = 3

    # dimensions
    n1, n2, n3 = len(x1), len(x2), len(x3)
    nc.createDimension('nx', n1)
    nc.createDimension('ny', n2)
    nc.createDimension('nz', n3)
    nc.createDimension('ndims', ndims)

    # grid
    nc_x = nc.createVariable('x', N.Float32, ('nx', 'ndims',))
    nc_y = nc.createVariable('y', N.Float32, ('ny', 'ndims',))
    nc_z = nc.createVariable('z', N.Float32, ('nz', 'ndims',))
    nc_x[:,0] = x1.astype(N.Float32); nc_x[:,1] = 0.                  ; nc_x[:,2] = 0.
    nc_y[:,0] = 0.                  ; nc_y[:,1] = x2.astype(N.Float32); nc_y[:,2] = 0.
    nc_z[:,0] = 0.                  ; nc_z[:,1] = 0.                  ; nc_z[:,2] = x3.astype(N.Float32)

    # field
    nc_field = nc.createVariable(name, N.Float32, ('nz', 'ny', 'nx', 'ndims',))
    nc_field.field = name + ', vector'
    nc_field.positions = 'z, product; y, product; x, product'
    nc_field[:,:,:,0] = field3.astype(N.Float32)
    nc_field[:,:,:,1] = field2.astype(N.Float32)
    nc_field[:,:,:,2] = field1.astype(N.Float32)

    nc.close()


def vectorField3_array(x1, x2, x3, field1, field2, field3, name):

    " (field1, field2, field3)(x1, x2, x3) -> name.dx file "

    f = file(name+'.dx', 'w')
    print >>f, '# OpenDX file '
    funct_name = sys._getframe().f_code.co_name
    print >>f, '# created by ', script_name, '::', funct_name, ' on ' + time.asctime()
    print >>f, '# 3-d positions'
    n1, n2, n3 = len(x1), len(x2), len(x3)
    print >>f, 'object 1 class array type float rank 1 shape 3 items %d data follows' \
          % (n1*n2*n3)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                print >>f, '%g %g %g'%(x1[i1],x2[i2],x3[i3])
    print >>f, '# regular connections'
    print >>f, 'object 2 class gridconnections counts %d %d %d'%(n1,n2,n3)
    print >>f, 'object 3 class array type float rank 1 shape 3 items %d data follows' \
          % (n1*n2*n3)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                print >>f, '%g %g %g' % (field1[i3, i2, i1], field2[i3, i2, i1], field3[i3, i2, i1])
    print >>f,"""#
object "irreg positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
#
end
"""


def vectorField2_array_netcdf(x1, x2, field1, field2, name):

    " (field1, field2, field3)(x1, x2, x3) -> name.nc file "

    nc = NetCDFFile(name+'.nc','w')

    ndims = 2

    # dimensions
    n1, n2 = len(x1), len(x2)
    nc.createDimension('nx', n1)
    nc.createDimension('ny', n2)
    nc.createDimension('ndims', ndims)

    # grid
    nc_x = nc.createVariable('x', N.Float32, ('nx', 'ndims',))
    nc_y = nc.createVariable('y', N.Float32, ('ny', 'ndims',))
    nc_x[:,0] = x1.astype(N.Float32); nc_x[:,1] = 0.
    nc_y[:,0] = 0.                  ; nc_y[:,1] = x2.astype(N.Float32)

    # field
    nc_field = nc.createVariable(name, N.Float32, ('ny', 'nx', 'ndims',))
    nc_field.field = name + ', vector'
    nc_field.positions = 'y, product; x, product'
    nc_field[:,:,0] = field2.astype(N.Float32)
    nc_field[:,:,1] = field1.astype(N.Float32)

    nc.close()

def vectorField2_array(x1, x2, field1, field2, name):

    " (field1, field2)(x1, x2) -> name.dx file "

    f = file(name+'.dx', 'w')
    print >>f, '# OpenDX file '
    funct_name = sys._getframe().f_code.co_name
    print >>f, '# created by ', script_name, '::', funct_name, ' on ' + time.asctime()
    print >>f, '# 2-d positions'
    n1, n2 = len(x1), len(x2)
    print >>f, 'object 1 class array type float rank 1 shape 2 items %d data follows' \
          % (n1*n2)
    for i1 in range(n1):
        for i2 in range(n2):
            print >>f, '%g %g'%(x1[i1],x2[i2])
    print >>f, '# regular connections'
    print >>f, 'object 2 class gridconnections counts %d %d'%(n1,n2)
    print >>f, 'object 3 class array type float rank 1 shape 2 items %d data follows' \
          % (n1*n2)
    for i1 in range(n1):
        for i2 in range(n2):
            print >>f, '%g %g' % (field1[i2, i1], field2[i2, i1])
    print >>f,"""#
object "irreg positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
#
end
"""

###############################################################################

if __name__=='__main__':

    import Numeric as N
    
    n1, n2, n3 = 20, 21, 23
    eps = 1.e-8
    x1 = N.arange(0., 1.+eps, 1./(n1-1))
    x2 = N.arange(0., 2.+eps, 2./(n2-1))
    x3 = N.arange(0., 3.+eps, 3./(n3-1))
    f = N.zeros((n3,n2,n1), N.Float64)
    v1 = N.zeros((n3,n2,n1), N.Float64)
    v2 = N.zeros((n3,n2,n1), N.Float64)
    v3 = N.zeros((n3,n2,n1), N.Float64)
    for i3 in range(n3):
        for i2 in range(n2):
            for i1 in range(n1):
                f[i3,i2,i1] = x1[i1]**2 + x2[i2]**2 + x3[i3]**2
                v1[i3, i2, i1] = N.cos(N.pi*x1[i1])
                v2[i3, i2, i1] = N.cos(N.pi*x2[i2]/2.)
                v3[i3, i2, i1] = N.sin(N.pi*x3[i3]/6.)
                
    scalarField3_array(x1, x2, x3, f, 'f3')
    scalarField3_array_netcdf(x1, x2, x3, f, 'f3')
    
    scalarField2_array(x1, x2, f[0,:,:], 'f2')
    scalarField2_array_netcdf(x1, x2, f[0,:,:], 'f2')
    
    vectorField3_array(x1,x2,x3, v1, v2, v3, 'v3')
    vectorField3_array_netcdf(x1,x2,x3, v1, v2, v3, 'v3')
    
    vectorField2_array(x1,x2, v1[0,:,:], v2[0,:,:], 'v2')
    vectorField2_array_netcdf(x1,x2, v1[0,:,:], v2[0,:,:], 'v2')
    
