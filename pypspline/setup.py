#!/usr/bin/env python

# $Id$

LIBS = ['pspline', 'ezcdf', 'portlib',]
INCS = ['./', ]

# the following should be edited to reflect your settings
###############################################################################

# f2py macros (see f2py doc)
MACROS = [] # [('F2PY_REPORT_ATEXIT', '1'),]

# if you plspline in another directory than ../LINUX/lib, change the line below
LIBLOC = ['../LINUX/lib',] # location of libpspline.a, etc.

# the following are Fortran libraries that the C compiler must link with
#
# Example 1: Intel fortran compiler ifort version 8.0 with gcc 3.3.2
# For the Intel compiler ifort version 8.0 these are:
LIBLOC += ['/opt/intel_fc_80/lib/',]
LIBS += ['ifcore',]

# Example 2: Intel fortran compiler ifort version 8.0 with gcc 2.96, and
##LIBLOC += ['/usr/local/intel/compiler80/intel_fc_80/lib/',]
##LIBS += ['ifcore', 'cxa', 'unwind',]
# to access Numpy's include files
##INCS += ['/home/ap/include/python2.2/']

# Example 3: LaheyFujitsu fortran compiler
##LIBLOC += ['/usr/local/lf95/lib/',]
##LIBS += ['fj9i6','fj9f6','fj9e6',]
# to access Numpy's include files
##INCS += ['/home/ap/include/python2.2/']

# Example 4: Portland compiler pg90 with gcc 2.96
##LIBLOC += ['/usr/local/pgi/linux86/lib',]
##LIBS += ['pgf90', 'pgf90_rpm1', 'pgf902', 'pgf90rtl', 'pgc', ]
##INCS += ['/home/ap/include/python2.2/']





###############################################################################

from distutils.core import setup, Extension

fpspline = Extension('fpspline',
                    define_macros = MACROS,
                    include_dirs = INCS,
                    library_dirs = LIBLOC,
                    libraries = LIBS,
                    sources = ['fpsplinemodule.c',
                               'fortranobject.c',])

setup (name = 'pypspline',
       version = '0.1',
       description = 'Spline interpolation in 1 to 3 dimensions',
       author_email = 'Alexander.Pletzer@noaa.gov',
       url = 'http://pypspline.sourceforge.net',
       long_description = '''
PyPSPLINE a python interface to the fortran spline library PSPLINE for
interpolating and computing derivatives of functions in 1 to 3
dimensions with control over boundary conditions.
''',
       ext_modules = [fpspline],
       py_modules = ['pspline3_r8',
                     'pspline3_r4',
                     ]
       )

