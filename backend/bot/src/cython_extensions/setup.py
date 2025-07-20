from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "information_set_fast",
        ["information_set_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    ),
    Extension(
        "game_logic_fast",
        ["game_logic_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    ),
    Extension(
        "cfr_fast",
        ["cfr_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    ),
    Extension(
        "blueprint_optimisation_fast",
        ["blueprint_optimisation_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'embedsignature': True
    }),
    zip_safe=False
)
