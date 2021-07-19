from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("filter",
                  sources=["filter.cpp"],
                  include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                  extra_compile_args=['-std=c++17'])

c_ext.language = 'c++'
setup(
    name='filter',
    version='1.0',
    description='Template Filter',
    ext_modules=[c_ext],
)