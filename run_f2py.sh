#!/bin/bash

python -m numpy.f2py -m pspline_wrapped -h src/pspline_wrapped.pyf --overwrite-signature \
    only: \
    genxpkg \
    mkspline evspline vecspline \
    mkbicub  evbicub  vecbicub  gridbicub  \
    mktricub evtricub vectricub gridtricub \
    : src/*.f90

