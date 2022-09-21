from numpy.distutils.core import Extension
import setuptools

pspline_ext = Extension(name = 'fpspline',
                        sources = ['src/fpspline.pyf',
                                   'src/cspline.f90',
                                   'src/evbicub.f90',
                                   'src/evspline.f90',
                                   'src/evtricub.f90',
                                   'src/genxpkg.f90',
                                   'src/gridbicub.f90',
                                   'src/gridtricub.f90',
                                   'src/herm1ev.f90',
                                   'src/herm2ev.f90',
                                   'src/herm3ev.f90',
                                   'src/ibc_ck.f90',
                                   'src/mkbicub.f90',
                                   'src/mkspline.f90',
                                   'src/mktricub.f90',
                                   'src/splinck.f90',
                                   'src/vecbicub.f90',
                                   'src/vecspline.f90',
                                   'src/vectricub.f90',
                                   'src/v_spline.f90',
                                   'src/xlookup.f90',
                                   'src/zonfind.f90'])

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'pypspline3',
          version ='1.0.0',
          author            = "Jonathan Schilling",
          author_email      = "jonathan.schilling@mail.de",
          description       = "Princeton Spline and Hermite Cubic Interpolation Routines ",
          long_description=long_description,
          long_description_content_type="text/markdown",
          url="https://github.com/jonathanschilling/pypspline3",
          packages = setuptools.find_packages(),
          ext_modules = [pspline_ext],
          classifiers=["Programming Language :: Python :: 3",
                       "Operating System :: POSIX :: Linux",
                       ],
          python_requires='>=3',
          install_requires=['numpy'],
          )

