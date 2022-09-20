
from pypspline.pspline_wrapped import genxpkg

import numpy as np

nx = 11
x = np.linspace(0.0, 1.0, nx)

xpkg = np.zeros([nx, 4], order='F')
#xpkg = np.zeros([nx, 4])

iper = 1
imsg = 1
itol = 0
ztol = 5.0e-7
ialg = -3
ier = 0

genxpkg(nx, x, xpkg, iper, imsg, itol, ztol, ialg, ier)

print(xpkg)

