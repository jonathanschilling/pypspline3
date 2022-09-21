#!/bin/bash

rm -rf obj/*
rm -f pypspline/*pspline_wrapped*
rm -f f90wrap_toplevel.f90

mkdir -p obj/

for i in src/*.f90
do
  gfortran -c ${i} -o obj/`echo ${i} | sed -e 's/^src//g' | sed -e 's/f90$/o/g'`
done

f90wrap -m pspline_wrapped -k kind_map src/*.f90

f2py -c -m _pspline_wrapped -DF2PY_REPORT_ON_ARRAY_COPY=1 f90wrap_toplevel.f90 obj/*.o

# for later, when import setup is fixed
#mv *pspline_wrapped* pypspline/

