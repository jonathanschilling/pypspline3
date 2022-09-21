#!/bin/bash

python -m numpy.f2py -m fpspline -h src/fpspline.pyf --overwrite-signature \
    only: \
    genxpkg \
    mkspline evspline vecspline \
    mkbicub  evbicub  vecbicub  gridbicub  \
    mktricub evtricub vectricub gridtricub \
    : src/*.f90

