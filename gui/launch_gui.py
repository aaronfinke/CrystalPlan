"""GUI launcher for CrystalPlan."""
#Boa:App:CrystalPlanApp

# Author: Janik Zikovsky, zikovskyjl@ornl.gov
# Version: $Id: CrystalPlan.py 1127 2010-04-01 19:28:43Z 8oz $
#print "CrystalPlan.gui.launch_gui is being imported. __name__ is", __name__
#print "CrystalPlan.gui.launch_gui __file__ is", __file__

#--- GUI Imports ---
import frame_main
import frame_qspace_view
import display_thread
import wx
import model
   


#-------------------------------------------------------------------------
# The following is generated by BOA Constructor IDE
modules ={u'detector_views': [0, '', u'detector_views.py'],
 u'dialog_preferences': [0, '', u'dialog_preferences.py'],
 u'dialog_startup': [0, '', u'dialog_startup.py'],
 u'display_thread': [0, '', u'display_thread.py'],
 u'experiment': [0, '', u'model/experiment.py'],
 u'frame_main': [1, 'Main frame of Application', u'frame_main.py'],
 u'frame_qspace_view': [0, '', u'frame_qspace_view.py'],
 u'frame_reflection_info': [0, '', u'frame_reflection_info.py'],
 u'frame_test': [0, '', u'frame_test.py'],
 u'goniometer': [0, '', u'model/goniometer.py'],
 u'gui_utils': [0, '', u'gui_utils.py'],
 u'instrument': [0, '', u'model/instrument.py'],
 u'messages': [0, '', u'model/messages.py'],
 u'panel_add_positions': [0, '', u'panel_add_positions.py'],
 u'panel_detectors': [0, '', u'panel_detectors.py'],
 u'panel_experiment': [0, '', u'panel_experiment.py'],
 u'panel_goniometer': [0, '', u'panel_goniometer.py'],
 u'panel_positions_select': [0, '', u'panel_positions_select.py'],
 u'panel_qspace_options': [0, '', u'panel_qspace_options.py'],
 u'panel_reflection_info': [0, '', u'panel_reflection_info.py'],
 u'panel_reflection_measurement': [0, '', u'panel_reflection_measurement.py'],
 u'panel_reflections_view_options': [0,'',u'panel_reflections_view_options.py'],
 u'panel_sample': [0, '', u'panel_sample.py'],
 u'plot_panel': [0, '', u'plot_panel.py'],
 u'scd_old_code': [0, '', u'scd_old_code.txt'],
 u'slice_panel': [0, '', u'slice_panel.py']}


#The background display thread
global background_display
background_display = None


#-------------------------------------------------------------------------
class CrystalPlanApp(wx.App):
    def OnInit(self):
        #Create the main GUI frame
        self.main = frame_main.create(None)
        self.main.Show()
        #Set it on top
        self.SetTopWindow(self.main)
        #Also, we show the q-space coverage window
        frame_qspace_view.get_instance(self.main).Show()
        return True


#-------------------------------------------------------------------------
#if __name__ == '__main__':
def launch_gui():
#    print "CrystalPlan GUI launching from", __file__
#    print "__name__ is", __name__
    
    #TODO: Here pick the latest instrument, load other configuration
    #Make the goniometers
    model.goniometer.initialize_goniometers()

    #Ok, create the instrument
    model.instrument.inst = model.instrument.Instrument(model.config.cfg.default_detector_filename)
    model.instrument.inst.make_qspace()

    #Initialize the instrument and experiment
    model.experiment.exp = model.experiment.Experiment(model.instrument.inst)
    model.experiment.exp.crystal.point_group_name = model.crystals.get_point_group_names(long_name=True)[0]

    #Some initial calculations
    if False:
        import numpy as np
        for i in np.deg2rad([-5, 0, 5]):
            model.instrument.inst.simulate_position(list([i,i,i]))
        pd = dict()
        for pos in model.instrument.inst.positions:
            pd[ id(pos) ] = True
        display_thread.NextParams[model.experiment.PARAM_POSITIONS] = model.experiment.ParamPositions(pd)
        #Do some reflections
        model.experiment.exp.initialize_reflections()
        model.experiment.exp.recalculate_reflections(model.experiment.ParamPositions(pd))
    else:
        model.experiment.exp.initialize_reflections()
        model.experiment.exp.recalculate_reflections(None)

    #Initialize the application
    application = CrystalPlanApp(0)

    #Start the thread that monitors changes to the q-space coverage calculation.
    if True: #To try it with and without
        background_display = display_thread.DisplayThread()
        display_thread.thread_exists = True
    else:
        display_thread.thread_exists = False

    #Start the GUI loop
    application.MainLoop()

    #Exit the program and do all necessary clean-up.
    print "Exiting CrystalPlan. Have a nice day!"
    background_display.abort()


if __name__=="__main__":
    #For launching from source
    launch_gui()