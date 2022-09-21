# PyPSPLINE
PyPSPLINE is a Python interface to the Fortran spline library [PSPLINE](https://w3.pppl.gov/ntcc/PSPLINE/).

The routines of the `compact` F77 interface are wrapped in Python classes.
Internal documentation of the underlying Fortran routines can be found in the [PSPLINE Help](https://w3.pppl.gov/~pshare/help/pspline.htm)
at [F77_standard_software](https://w3.pppl.gov/~pshare/help/body_pspline_hlp.html#outfile24.html) --> [Compact_Splines](https://w3.pppl.gov/~pshare/help/body_pspline_hlp.html#outfile30.html) .

## Prerequisites

* Python 3 (tested with 3.10.7): http://python.org
* NumPy: http://numpy.org/

A Fortran 90 and a C compiler.

I assume the platform to be UNIX. If you manage to build PSPLINE/pypspline on Windows, let me know.

## Building PyPSPLINE

A minimal subset of PSPLINE required for 1D, 2D and 3D spline interpolation with periodic boundary conditions
is included in this repository.

The setup relies on `numpy.disttools` as suggested in the [`f2py` documentation](https://numpy.org/doc/stable/f2py/buildtools/distutils.html).
This will likely shift to `meson` once SciPy and NumPy have migrated as well [1](https://github.com/scipy/scipy/issues/13615).

For now, the build and installation process goes as follows:

```bash
pip install --user .
```

The `f2py` signature file [`src/fpyspline.pyf`](src/fpyspline.pyf) was auto-generated using the included [`run_f2py.sh`](run_f2py.sh) script
and then hand-adjusted to line up with the assumptions made in [the original](https://github.com/jonathanschilling/pypspline/blob/ab3a6858cb77345be1403be16061a27efdcd91a2/pypspline/fpspline/fpspline.pyf).

## History

This package was originally created by Alexander Pletzer
and published on SourceForce as [`pypspline`](https://sourceforge.net/projects/pypspline/).

The old [PyPSPLINE CVS repository](https://sourceforge.net/projects/pypspline/) was migrated to this Git repository.
One needs the `cvs` and `cvs2svn` packages to do this on Arch Linux.

```bash
rsync -ai a.cvs.sourceforge.net::cvsroot/pypspline/ cvs2git-pypspline
cd cvs2git-pypspline
cvs2git --blobfile=blob.dat --dumpfile=dump.dat \
    --username=pletzer --default-eol=native \
    --encoding=utf8 --encoding=latin1 --fallback-encoding=ascii \
    .
cd ..
mkdir pypspline
cd pypspline
git init
cat ../cvs2git-pypspline/blob.dat ../cvs2git-pypspline/dump.dat | git fast-import
git remote add origin git@github.com:jonathanschilling/pypspline.git
git checkout
git push origin --mirror
mv pypspline{,_old}
mv pypspline_old/* .
rm -r CVSROOT
git add .
git commit -m "one folder less"
git branch --set-upstream-to=origin/master master
git pull
git push
```
