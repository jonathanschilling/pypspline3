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

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'pypspline',
          version ='1.0.0',
          description       = "Princeton Spline and Hermite Cubic Interpolation Routines ",
          author            = "Jonathan Schilling",
          author_email      = "jonathan.schilling@mail.de",
          packages = setuptools.find_packages(),
          ext_modules = [pspline_ext]
          )

