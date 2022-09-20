#!/usr/bin/env python

print("fpspline being imported")

import ctypes
import numpy as np

libpspline = ctypes.cdll.LoadLibrary("libpspline.so")

def genxpkg(x, iper):

    c_iper = ctypes.c_int(iper)

    nx = ctypes.c_int(len(x))

    c_x_type = ctypes.c_float * nx.value
    c_x = np.ctypeslib.as_ctypes(np.ascontiguousarray(x.T))

    xpkg = np.empty(nx.value * 4, dtype=np.float32)
    c_xpkg = np.ctypeslib.as_ctypes(np.ascontiguousarray(xpkg.T))

    c_imsg = ctypes.c_int(1)
    c_itol = ctypes.c_int(0)
    c_ztol = ctypes.c_float(5.0e-7)

    c_ialg = ctypes.c_int(-3)
    c_ier  = ctypes.c_int(0)

    libpspline.genxpkg_(nx, c_x, c_xpkg, c_iper,
                        c_imsg, c_itol, c_ztol,
                        c_ialg, c_ier)

    return c_xpkg, c_ier.value

def mkspline(x, flat, ibctype0, bcval1min, ibctype1, bcval1max):

    ilin1 = 1
    ifail = 0

    return ilin1, ifail

def evspline(p, x, ilin1, flat, ictFval):

    fi = None
    ier = 0

    return fi, ier

def vecspline(ictFval, p, x1pkg, flat):

    fi = None
    iwarn = 0
    ier = 0

    return fi, iwarn, ier
