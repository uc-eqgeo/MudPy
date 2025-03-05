'''
Fakequakes runs with a simple planar geometry
'''


from mudpy import fakequakes, runslip, forward
import numpy as np
from obspy.core import UTCDateTime
import os


########                            GLOBALS                             ########
home = os.getcwd() + '/examples/fakequakes/planar/'
project_name = 'hikkerk'
run_name = 'subduction'
################################################################################


##############             What do you want to do??           ##################
init = 1
make_ruptures = 1
make_GFs = 0
make_synthetics = 0
make_waveforms = 0
# Things that only need to be done once
load_distances = 1
G_from_file = 0
###############################################################################


#############                 Run-time parameters            ##################

ncpus = 14  # only useful for waveform synthesis

model_name = 'bbp_norcal_swapPS.mod'   # Velocity model
fault_name = 'subduction.fault'    # Fault geometry
slab_name = None   # Slab 1.0 Ascii file (only used for 3D fault)
mesh_name = None    # GMSH output file (only used for 3D fault)
distances_name = 'planar_subduction'  # Name of distance matrix
# Don't change this (unless you know waht you're doing!)
rupture_list = 'ruptures.list'
# Look here if unsure (https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#/media/File:Utm-zones.jpg)
UTM_zone = '31N'
scaling_law = 'T'  # T for thrust, S for strike-slip, N for normal


# slip parameters
Nrealizations = 5  # Number of fake ruptures to generate per magnitude bin
target_Mw = np.arange(7.5, 8.4, 0.1)  # Of what approximate magnitudes
max_slip = 60  # Maximum slip (m) allowed in the model

# Correlation function parameters
hurst = 0.75
Ldip = 'auto'      # Correlation length scaling, 'auto' uses  Mai & Beroza 2002,
Lstrike = 'auto'   # MH2019 uses Melgar & Hayes 2019
lognormal = True
slip_standard_deviation = 0.9
num_modes = 500
rake = 90.0

# Rupture parameters
force_magnitude = True  # Make the magnitudes EXACTLY the value in target_Mw
# Forces using the entire fault area defined by the .fault file as opposed to the scaling laws
force_area = False
no_random = False  # If true uses median length/width if false draws from prob. distribution
time_epi = UTCDateTime('2016-09-07T14:42:26')  # Defines the hypocentral time
# Defines the specific hypocenter location if force_hypocenter=True
hypocenter = [0.8301, 0.01, 27.67]
# Forces hypocenter to occur at specified lcoationa s opposed to random
force_hypocenter = False
mean_slip = None  # Provide path to file name of .rupt to be used as mean slip pattern
# Integer value, if != None use that subfault as center for defining rupt area. If none then slected at random
center_subfault = None
# If true use hypocenter PDF positions from Melgar & Hayes 2019, if false then selects at random
use_hypo_fraction = False

# Kinematic parameters
source_time_function = 'dreger'  # options are 'triangle' or 'cosine' or 'dreger'
rise_time = 'MH2017'
rise_time_depths = [10, 15]  # Transition depths for rise time scaling
buffer_factor = 0.5   # idon't think this does anything anymore, remove?
# Fraction of shear wave speed to use as mean rupture velocity
shear_wave_fraction = 0.8

# Station information (only used when syntehsizing waveforms)
GF_list = 'coastal_gps.gflist'
G_name = 'GFs'

# Displacement and velocity waveform parameters
NFFT = 128
dt = 1.0

# fk-parameters
dk = 0.1
pmin = 0
pmax = 1
kmax = 20
custom_stf = None

###############################################################################


# Initalize project folders
if init == 1:
    fakequakes.init(home, project_name)

# Generate rupture models
if make_ruptures == 1:
    fakequakes.generate_ruptures(home, project_name, run_name, fault_name, slab_name,
                                 mesh_name, load_distances, distances_name, UTM_zone, target_Mw, model_name, hurst, Ldip,
                                 Lstrike, num_modes, Nrealizations, rake, rise_time, rise_time_depths, time_epi,
                                 max_slip, source_time_function, lognormal, slip_standard_deviation, scaling_law, ncpus,
                                 force_magnitude=force_magnitude, force_area=force_area, mean_slip_name=mean_slip,
                                 hypocenter=hypocenter, force_hypocenter=force_hypocenter, no_random=no_random,
                                 use_hypo_fraction=use_hypo_fraction)


# Prepare waveforms and synthetics
if make_GFs == 1 or make_synthetics == 1:
    runslip.inversionGFs(home, project_name, GF_list, None, fault_name, model_name,
                         dt, None, NFFT, None, make_GFs, make_synthetics, dk, pmin,
                         pmax, kmax, 0, time_epi, 0, ncpus, custom_stf, impulse=True)

# Synthesize the waveforms
if make_waveforms == 1:
    forward.waveforms_fakequakes(home, project_name, fault_name, rupture_list, GF_list,
                                 model_name, run_name, dt, NFFT, G_from_file, G_name, source_time_function)
