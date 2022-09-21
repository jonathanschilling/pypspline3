import numpy as np
import sys, time

from pypspline3.pspline_1d import pspline

import matplotlib.pyplot as plt


n1 = 11
bcs1 = 1
x1min, x1max = 0.0, 2.0*np.pi
x1 = np.linspace(x1min, x1max, n1)

f = np.cos(x1)

plt.figure()
plt.plot(x1, f, 'o-', label="orig")


spl = pspline(x1, bcs1=bcs1)
spl.setup(f)

n2 = 41
x2 = np.linspace(-np.pi, 3.0*np.pi, n2)
ffi, ier, iwarn = spl.interp_array(x2)

n3 = 301
x3 = np.linspace(-np.pi, 3.0*np.pi, n3)
fi = np.cos(x3)

plt.plot(x3, fi, '--', label="exact")
plt.plot(x2, ffi, 'x', label="interp")

plt.legend(loc="upper right")
plt.show()
