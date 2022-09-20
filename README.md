# PyPSPLINE
PyPSPLINE is a Python interface to the Fortran spline library [PSPLINE](https://w3.pppl.gov/ntcc/PSPLINE/).

## Prerequisites

* Python (tested with 3.10.7): http://python.org
* NumPy: http://numpy.org/
* f90wrap: https://github.com/jameskermode/f90wrap

A Fortran 90 and a C compiler.

I assume the platform to be UNIX. If you manage to build PSPLINE/pypspline on Windows, let me know. 

## Building PyPSPLINE

A minimal subset of PSPLINE is included in this repository.
Is is compiled along with auto-generating the Python wrapper code by `make.sh`.
Just run this script to compile the PSPLINE code using `gfortran`
and generate the Python wrapper using `f90wrap` and `f2py`:

```bash
./make.sh
```
## Migration from CVS

Created Sat Mar 13 08:31:22 EST 2004 (alexander@gokliya.net)

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
