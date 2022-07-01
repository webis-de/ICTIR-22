from setuptools import setup, Extension
#import cython_gsl

from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension(
        '_bradleyterry', sources=['_bradleyterry.pyx'],
        #libraries=cython_gsl.get_libraries(),
        #library_dirs=[cython_gsl.get_library_dir()],
        #include_dirs=[cython_gsl.get_cython_include_dir()]
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
    install_requires=[
        'cython'
        'cythongsl',
    ],
)
