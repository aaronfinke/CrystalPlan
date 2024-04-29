#!/usr/bin/env python
"""Script for installing the CrystalPlan utility."""

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#--- Imports ---
from setuptools import setup, find_packages, Extension
import sys
import CrystalPlan_version
from Cython.Build import cythonize
import numpy


#Two packages: the GUI and the model code
packages = find_packages()
print(packages)
packages = ['CrystalPlan', 'CrystalPlan.model',  'CrystalPlan.pyevolve', 'CrystalPlan.gui', 'CrystalPlan.model.pygene','CrystalPlan.model.cython_routines']
package_dir = {'CrystalPlan': '.',  'CrystalPlan.pyevolve':'pyevolve', 'CrystalPlan.model':'model', 'CrystalPlan.gui': 'gui', 'CrystalPlan.model.pygene':'model/pygene',
               'CrystalPlan.model.cython_routines': 'model/cython_routines'}
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

def pythonVersionCheck():
    # Minimum version of Python
    PYTHON_MAJOR = 2
    PYTHON_MINOR = 5

    if sys.version_info < (PYTHON_MAJOR, PYTHON_MINOR):
        print('You need at least Python %d.%d for %s %s' \
              % (PYTHON_MAJOR, PYTHON_MINOR, CrystalPlan_version.package_name, CrystalPlan_version.version), file=sys.stderr)
        sys.exit(-3)

if __name__ == "__main__":
    pythonVersionCheck()
    
    ext_modules = [
        Extension(
            "cython_routines",
            ["model/cython_routines/*.pyx"],
            extra_compile_args=['-Xclang','-fopenmp'],
            extra_link_args=['-Xclang','-fopenmp']
        )
    ]

    setup(name=CrystalPlan_version.package_name,
          version=CrystalPlan_version.version,
          description=CrystalPlan_version.description,
          author=CrystalPlan_version.author, author_email=CrystalPlan_version.author_email,
          url=CrystalPlan_version.url,
          scripts=scripts,
          packages=packages,
          package_dir=package_dir,
          data_files=data_files,
          package_data=package_data,
          #include_package_data=True,
          install_requires=install_requires,
          include_dirs=[numpy.get_include()],
    #test_suite='model.test_all.get_all_tests'
          ext_modules=cythonize(ext_modules, annotate=True, 
          )
    )
