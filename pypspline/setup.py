#!/usr/bin/env python

# $Id$

LIBS = ['pspline', 'ezcdf', 'portlib',]

# the following should be edited to reflect your settings
###############################################################################

MACROS = [('F2PY_REPORT_ATEXIT', '1'),]

# if you plspline in another directory than ../LINUX/lib, change the line below
LIBLOC = ['../LINUX/lib',] # location of libpspline.a, libezcdf.a & libportlib.a

# the following are Fortran libraries that the C compiler must link with
# For the Intel compiler ifort version 8.0 these are:
LIBLOC += ['/opt/intel_fc_80/lib/',]
LIBS += ['ifcore',]

###############################################################################

from distutils.core import setup, Extension

fpspline = Extension('fpspline',
                    define_macros = MACROS,
                    include_dirs = ['./',],
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

