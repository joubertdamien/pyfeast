#!/bin/sh
rm pyfeast.cpython-37m-x86_64-linux-gnu.so
python setup.py build_ext --inplace
python setup.py install
python test_filter.py 

