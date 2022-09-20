#!/bin/bash

rm -rf obj/*
rm pypspline/*pspline_wrapped*

mkdir -p obj/

f90wrap -m pspline_wrapped -k kind_map src/*.f90

for i in src/*.f90
do
  gfortran -c ${i} -o obj/`echo ${i} | sed -e 's/^src//g' | sed -e 's/f90$/o/g'`
done

f2py -c -m _pspline_wrapped -DF2PY_REPORT_ON_ARRAY_COPY=1 f90wrap_toplevel.f90 obj/*.o

mv *pspline_wrapped* pypspline/

