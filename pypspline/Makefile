# -*-Makefile-*-

# $Id$

MODULE = fpspline

.SUFFIXES: .so

$(MODULE)module.c: $(MODULE).pyf
	f2py $<



all: $(MODULE).so

realclean:
	rm -rf *.so