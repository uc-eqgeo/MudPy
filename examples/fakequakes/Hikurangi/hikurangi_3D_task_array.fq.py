'''
Parameter file for 3D fakequakes run
'''


from mudpy import fakequakes,runslip,forward
import numpy as np
from obspy.core import UTCDateTime
import argparse
import os



########                            GLOBALS                             ########
home=os.path.join('/nesi', 'nobackup', 'uc03610', 'jack', 'fakequakes') + os.sep
project_name='hikkerm' # Directory name
run_name='hikkerm' # Name for this run

######### Fixing run paths to work with linux or windows ###########
if 'mnt' in os.getcwd().split(os.sep)[:2]:
    root = home.split(':')[0].lower() + os.sep
    home = '/mnt/' + root + (os.sep).join(home.split(os.sep)[1:])

################################################################################


##############             What do you want to do??           ##################
# Things that only need to be done once
load_distances=1
###############################################################################


#############                 Run-time parameters            ##################

#######  OCC Parameters #######
ncpus=1
model_name='wuatom.mu'   # Velocity model. .mod for layered velocity, .mu for 3D rigidity
fault_name='hk.fault'
UTM_zone='60'
scaling_law='T' # T for thrust, S for strike-slip, N for normal

# Correlation function parameters
NZNSHM_scaling = True # Enforce New Zealand NSHM scaling law of Mw = log10(area) + 4.0

# Rupture parameters
hypocenter=None #=None is random hypocenter
rake='vary' # average rake, or 'vary' for variable rake based off fault model
mean_slip_name = 'hk_lock.slip'  # Variable that contains the mean slip distribution (i.e. slip deficit model) - full file path (Needs to be in .rupt format)
uniform_slip=False # If true, skip the stochastic aspect of this whole process and just use relatively uniform slip based on velocity model (equivialent to VAUS of Davies 2019)
sub_fault_end=6233  # Max patch number to nucleate faults on (-1 for all patches)

#Enforcement of rules on area scaling and hypo location
force_area=False
force_magnitude=True

#######  FakeQuakes defaults #######
slab_name=None    # Slab 1.0 Ascii file
mesh_name='hikkerm.mshout'  
distances_name=fault_name # Name of distances matrices

max_slip=100 #Maximum sip (m) allowed in the model
max_slip_rule=True #restrict max slip to 3 times Allen & Hayes 2017

# Correlation function parameters
hurst=0.4 # Melgar and Hayes 2019 found Hurst exponent is probably closer to 0.4
Ldip='auto' # Correlation length scaling, 'auto' uses Melgar & Hayes 2019
Lstrike='auto' # MB2002 uses Mai & Beroza 2002
lognormal=True # Keep this as true
slip_standard_deviation=0.46 # Value from Melgar & Hayes 2019

# Rupture parameters
time_epi=UTCDateTime('2014-04-01T23:46:47Z')
source_time_function='dreger' # options are 'triangle' or 'cosine' or 'dreger'
num_modes=200 # The more modes, the better you can model the high frequency stuff
rake='vary' # average rake, or 'vary' for variable rake based off fault model
rise_time = 'MH2017'
rise_time_depths=[10,15] #Transition depths for rise time scaling (if slip shallower than first index, rise times are twice as long as calculated)
calculate_rupture_onset=False # Calcualte rupture onset times. Slow, and useful for some applications, but not really for just generating ruptures

#Enforcement of rules on area scaling and hypo location
force_hypocenter=False
use_hypo_fraction=False
###############################################################################

#Initalize project folders
project_dir = os.path.join(home, project_name)
if not os.path.exists(project_dir):
    os.makedirs(project_dir)
    os.makedirs(os.path.join(project_dir, 'data', 'distances'))
    os.makedirs(os.path.join(project_dir, 'data', 'model_info'))
    os.makedirs(os.path.join(project_dir, 'structure'))
    raise Exception(f'Project directory {project_dir} created.\nPlease fill with model info')

if not os.path.exists(os.path.join(project_dir, 'output', 'ruptures')):
        os.makedirs(os.path.join(project_dir, 'output', 'ruptures'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--task_number", type=int, default=0, help="Task number for the SLURM array")
    parser.add_argument("--task_file", type=str, default="task_arrays.txt", help="File containing task array information")

    array_list = parser.parse_args().task_file
    with open(array_list, 'r') as f:
        lines = f.readlines()
        line = lines[parser.parse_args().task_number]

    _, min_mw, max_mw, mw_step, Nstart, n_rupt, to_make = line.strip('\n').split(',')
    print(f"{to_make} ruptures expected to be made")

    Nrealizations= int(n_rupt) # Number of fake ruptures to generate per magnitude bin
    target_Mw=np.round(np.arange(float(min_mw),float(max_mw),float(mw_step)),4) # Of what approximate magnitudes

    #Generate rupture models
    if mean_slip_name is None:
        tag = '_noMeanSlip'
    else:
        tag = f"_{mean_slip_name.replace('.slip', '').replace('hk_', '')}"
        mean_slip_name = os.path.join(home, project_name, 'data', 'model_info', mean_slip_name) # Variable that contains the mean slip distribution (i.e. slip deficit model) - full file path (Needs to be in .rupt format)

    tag += f'_{'.'.join([name_part for name_part in model_name.split('.')[:-1]])}'

    if NZNSHM_scaling:
        tag += '_NSHMarea'
    else:
        tag += '_noNSHMarea'

    if uniform_slip:
        stochastic_slip = False
        tag += '_uniformSlip'
    else:
        stochastic_slip = True

    run_name += tag
    fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,
            mesh_name,load_distances,distances_name,UTM_zone,target_Mw,model_name,
            hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,rise_time,
            rise_time_depths,time_epi,max_slip,source_time_function,lognormal,
            slip_standard_deviation,scaling_law,ncpus,mean_slip_name=mean_slip_name,
            force_magnitude=force_magnitude,force_area=force_area,hypocenter=hypocenter,
            force_hypocenter=force_hypocenter,
            max_slip_rule=max_slip_rule,use_hypo_fraction=use_hypo_fraction, 
            calculate_rupture_onset=calculate_rupture_onset, NZNSHM_scaling=NZNSHM_scaling,
            stochastic_slip=stochastic_slip, sub_fault_end=sub_fault_end,Nstart=int(Nstart))