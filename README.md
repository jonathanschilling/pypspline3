# PyPSPLINE
PyPSPLINE is a Python interface to the Fortran spline library PSPLINE. 

## Prerequisites

* Python (tested for 2.2 and 2.3): http://python.org
* NetCDF http://www.unidata.ucar.edu/packages/netcdf/
* NumPy: http://numpy.org/
* [optionally f2py: http://cens.ioc.ee/projects/f2py2e/]

A Fortran 90 and a C compiler.

I assume the platform to be UNIX. If you manage to build PSPLINE/pypspline on Windows, let me know. 

## Building PSPLINE

You will need to build PSPLINE first. Download the code from 

```bash
wget http://w3.pppl.gov/rib/repositories/NTCC/files/pspline.tar.gz
tar xvfz pspline.tar.gz
```

You should get the directories `pspline`, `ezcdf`, `portlib`, `include`, `fpreproc` and `share`. Be sure to set

```bash
export FORTRAN_VARIANT=<your_compiler>
```

if using `bash` shell, or

```csh
setenv FORTRAN_VARIANT <your_compiler>
```

if using `csh`.
Here, `<your_compiler>` can be `GCC`, `Intel`, `LaheyFujitsu`, `Portland`, `SUN`, `IRIX64`, etc.
`GCC` (meaning gfortran) works fine on Linux. 

If you're using an unsupported compiler you may want to add an entry to the file `share/Make.flags`. 

If NetCDF is not in the standard location (`/usr/local/`) you may need to edit `share/Make.local` or specify
via the `NETCDF_DIR` variable. Example:

```bash
make all FORTRAN_VARIANT=GCC NETCDF_DIR=/home/research/pletzer/software/netcdf
```

This will create a directory `LINUX/lib`
or `SUN/lib`, `SGI/lib`, ... depending on your platform.
Compilers are not discriminated so you will need to compile in different directories
if you want to support several compilers, say on Linux.
This is also true if you need to support both i686 and ia64.

## Building PyPSPLINE

I recommend downloading pyspline *after* PSPLINE in a directory adjacent to `pspline`, `ezcdf`, etc
(i.e. there should be no `pypspline` directory at his level when building PSPLINE). 

Go to this directory

```bash
cd pypspline
```

If you have a recent Python version installed on your system, you should not have to re-generate `fpsplinemodule.c`.

Edit the top lines of the `setup.py` file.
In particular, you will need to determine the Fortran libraries to provide to the C compiler, which are system dependent.
Pypspline has been successfully built on Linux using the Intel, Lahey-Fujitsu and Portland Group compilers. 

Then type

```bash
python setup.py install
```

or if you don't have root permission

```bash
python setup.py install --prefix=<path>
```

Note: on some systems `python` is called `python2`.

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
