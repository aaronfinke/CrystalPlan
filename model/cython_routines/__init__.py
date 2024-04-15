"""__init__.py module setup file."""
# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id$

#print "CrystalPlan.model being imported from", __file__
#print "CrystalPlan.model, __name__ is", __name__

#We import all the modules in this so as to make "import model" do all the necessary work.
from CrystalPlan.model.cython_routines import point_search
