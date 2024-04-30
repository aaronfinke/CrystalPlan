#!/usr/bin/env python
"""Script for installing the CrystalPlan utility."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Imports ---
from setuptools import setup, find_packages, Extension
import sys
# import CrystalPlan_version
from Cython.Build import cythonize
import numpy


#Two packages: the GUI and the model code
# packages = find_packages("src")
# print(packages)
# packages = ['CrystalPlan', 'CrystalPlan.model',  'CrystalPlan.pyevolve', 'CrystalPlan.gui', 'CrystalPlan.model.pygene','CrystalPlan.model.cython_routines']
# package_dir = {'CrystalPlan': '.',  'CrystalPlan.pyevolve':'pyevolve', 'CrystalPlan.model':'model', 'CrystalPlan.gui': 'gui', 'CrystalPlan.model.pygene':'model/pygene',
            #    'CrystalPlan.model.cython_routines': 'model/cython_routines'}
# data_files = [ ('instruments', './instruments/*.csv'), ('instruments', './instruments/*.xls') ]
data_files = []
package_data = {'CrystalPlan':['instruments/*.xls', 'instruments/*.csv', 'instruments/*.detcal',
                               'docs/*.*', 'docs/animations/*.*', 'docs/eq/*.*', 'docs/screenshots/*.*' ],
    'CrystalPlan.model':['data/*.*'],
    'CrystalPlan.gui':['icons/*.png']
}
scripts = ['crystalplan.py']

#Package requirements
install_requires = ['Traits', 'Mayavi', 'numpy', 'scipy', 'Cython']

if __name__ == "__main__":
    
    ext_modules = [
        Extension(
            "crystal_calcs",
            ["src/CrystalPlan/model/cython_routines/crystal_calcs.pyx"],
            extra_compile_args=['-Xclang','-fopenmp'],
            extra_link_args=['-Xclang','-fopenmp']
        ),
        Extension(
            "experiment_calcs",
            ["src/CrystalPlan/model/cython_routines/experiment_calcs.pyx"],
            extra_compile_args=['-Xclang','-fopenmp'],
            extra_link_args=['-Xclang','-fopenmp']
        ),
        Extension(
            "goniometers",
            ["src/CrystalPlan/model/cython_routines/goniometers.pyx"],
            extra_compile_args=['-Xclang','-fopenmp'],
            extra_link_args=['-Xclang','-fopenmp']
        ),
        Extension(
            "point_search",
            ["src/CrystalPlan/model/cython_routines/point_search.pyx"],
            extra_compile_args=['-Xclang','-fopenmp'],
            extra_link_args=['-Xclang','-fopenmp']
        ),
    ]

    setup(python_requires='>3.9',
          name="CrystalPlan",
          version="1.5",
          description="CrystalPlan is an experiment planning tool for crystallography. You can choose an instrument and supply your sample's lattice parameters to simulate which reflections will be measured, by which detectors and at what wavelengths.",
          author="Janik Zikovsky", author_email="zikovskyjl@ornl.gov",
          url="http://neutronsr.us",
          scripts=scripts,
        #   packages=packages,
        #   package_dir=package_dir,
          data_files=data_files,
          package_data=package_data,
          #include_package_data=True,
          install_requires=install_requires,
          include_dirs=['.',numpy.get_include()],
    #test_suite='model.test_all.get_all_tests'
          ext_modules=cythonize(ext_modules, annotate=True, 
          )
    )
