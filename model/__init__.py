"""__init__.py module setup file."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#print "CrystalPlan.model being imported from", __file__
#print "CrystalPlan.model, __name__ is", __name__

#We import all the modules in this so as to make "import model" do all the necessary work.
from CrystalPlan.model import config
from CrystalPlan.model import crystal_calc
from CrystalPlan.model import crystals
from CrystalPlan.model import detectors
from CrystalPlan.model import experiment
from CrystalPlan.model import goniometer
from CrystalPlan.model import instrument
from CrystalPlan.model import messages
from CrystalPlan.model import numpy_utils
from CrystalPlan.model import optimize_coverage
from CrystalPlan.model import reflections
from CrystalPlan.model import ubmatrixreader
from CrystalPlan.model import optimization
from CrystalPlan.model import utils
from CrystalPlan.model import tools
import CrystalPlan.model.cython_routines

