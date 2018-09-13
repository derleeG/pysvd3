from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import pkgconfig


extension = [
        Extension('svd3',
            ['svd3.pyx'],
            language='c++',
            include_dirs = [numpy.get_include()])]

setup(ext_modules = cythonize(extension))
