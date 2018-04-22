from distutils.core import setup, Extension

# define the extension module
pylinelib = Extension('pylinelib',
                       sources=['linelib.cpp', 'pylinelib.cpp'],
                       depends=['linelib.h', 'gsl/gsl_rng.h'],
                       include_dirs = ['/usr/local/include', '/Users/QuMn/research/software/eigen-3.2.4'],
                       library_dirs = ['/usr/local/lib'],
                       libraries=['gsl', 'gslcblas'],
                       extra_compile_args=['-lgsl -lm -lgslcblas -O3 -ffast-math'])

# run the setup
setup(ext_modules=[pylinelib])
