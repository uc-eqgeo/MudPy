'''
Parameter file for 3D fakequakes run
'''


from mudpy import fakequakes,runslip,forward
import numpy as np
from obspy.core import UTCDateTime
import argparse



########                            GLOBALS                             ########
home='/nesi/nobackup/uc03610/jack/fakequakes/' 
project_name='hikkerk3d_hires' # Directory name
run_name='hikkerk3D' # Name for this run
run_base_name='hikkerk3D'
################################################################################


##############             What do you want to do??           ##################
init=0
make_ruptures=1
make_GFs=0
make_synthetics=0
make_waveforms=0
make_hf_waveforms=0
match_filter=0
make_statics=0
# Things that only need to be done once
load_distances=1
G_from_file=0
###############################################################################


#############                 Run-time parameters            ##################
ncpus=1
hot_start=0
model_name='hikkerk.mod'   # Velocity model
moho_depth_in_km=25.0
fault_name='hk.fault'
slab_name='hk.slab'    # Slab 1.0 Ascii file
mesh_name='hikkerk.mshout'  
distances_name=fault_name # Name of distances matrices
rupture_list='ruptures.list'
UTM_zone='60'
scaling_law='T' # T for thrust, S for strike-slip, N for normal

#Station information
GF_list='hikkerk_gnss.gflist'
G_name=run_name  #Name of G matrix for waveforms
G_name_static=run_name+'_statics' #Name of G matrix for statics

##Nrealizations=125 # Number of fake ruptures to generate per magnitude bin
##target_Mw=np.round(np.arange(8.0,8.5,0.05),4) # Of what approximate magnitudes
max_slip=100 #Maximum sip (m) allowed in the model
max_slip_rule=True #restrict max slip to 3 times Allen & Hayes 2017

# Displacement and velocity waveform parameters
NFFT=128 ; dt=1.0
#fk-parameters
dk=0.1 ; pmin=0 ; pmax=1 ; kmax=20
custom_stf=None

#High frequency waveform parameters
hf_dt=0.01
duration=120
Pwave=True

#Match filter parameters
zero_phase=False
order=4
fcorner=1.0

# Correlation function parameters
hurst=0.4 # Melgar and Hayes 2019 found Hurst exponent is probably closer to 0.4
Ldip='auto' # Correlation length scaling, 'auto' uses Melgar & Hayes 2019
Lstrike='auto' # MB2002 uses Mai & Beroza 2002
NZNSHM_scaling = True # Enforce New Zealand NSHM scaling law of Mw = log10(area) + 4.0
lognormal=True # Keep this as true
slip_standard_deviation=0.46 # Value from Melgar & Hayes 2019

# Rupture parameters
time_epi=UTCDateTime('2014-04-01T23:46:47Z')
hypocenter=None #=None is random hypocenter
source_time_function='dreger' # options are 'triangle' or 'cosine' or 'dreger'
stf_falloff_rate=4 #Only affects Dreger STF, choose 4-8 are reasonaclosble values
num_modes=200 # The more modes, the better you can model the high frequency stuff
stress_parameter=50 #measured in bars
high_stress_depth=30 # SMGA must be below this depth (measured in km)
rake='vary' # average rake, or 'vary' for variable rake based off fault model
rise_time = 'MH2017'
rise_time_depths=[10,15] #Transition depths for rise time scaling (if slip shallower than first index, rise times are twice as long as calculated)
mean_slip_name=home+project_name+'/data/model_info/'+'hk_hires.slip'  # Variable that contains the mean slip distribution (i.e. slip deficit model) - full file path (Needs to be in .rupt format)
#mean_slip_name=None
shear_wave_fraction=0.8
calculate_rupture_onset=False # Calcualte rupture onset times. Slow, and useful for some applications, but not really for just generating ruptures

#Enforcement of rules on area scaling and hypo location
force_area=False
force_magnitude=True
force_hypocenter=False
use_hypo_fraction=False
###############################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--task_number", type=int, default=0, help="Task number for the SLURM array")


    array_list = 'task_arrays.txt'
    with open(array_list, 'r') as f:
        lines = f.readlines()
        line = lines[parser.parse_args().task_number]

    _, min_mw, max_mw, mw_step, Nstart, n_rupt = line.split(',')
    Nrealizations= int(n_rupt) # Number of fake ruptures to generate per magnitude bin
    target_Mw=np.round(np.arange(float(min_mw),float(max_mw),float(mw_step)),4) # Of what approximate magnitudes

    #Initalize project folders
    if init==1:
        fakequakes.init(home,project_name)
        
    #Generate rupture models
    if make_ruptures==1:
        mean_slip_name_list = [mean_slip_name]
        NZNSHM_scaling_list = [True]
        for mean_slip_name in mean_slip_name_list:
            for NZNSHM_scaling in NZNSHM_scaling_list:
                if mean_slip_name is None:
                    tag = '_nolocking'
                else:
                    tag = '_locking'
                if NZNSHM_scaling:
                    tag += '_NZNSHMscaling'
                else:
                    tag += '_noNZNSHMscaling'
                run_name = run_base_name + tag
                fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,
                        mesh_name,load_distances,distances_name,UTM_zone,target_Mw,model_name,
                        hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,rise_time,
                        rise_time_depths,time_epi,max_slip,source_time_function,lognormal,
                        slip_standard_deviation,scaling_law,ncpus,mean_slip_name=mean_slip_name,
                        force_magnitude=force_magnitude,force_area=force_area,hypocenter=hypocenter,
                        force_hypocenter=force_hypocenter,
                        max_slip_rule=max_slip_rule,use_hypo_fraction=use_hypo_fraction, 
                        calculate_rupture_onset=calculate_rupture_onset, NZNSHM_scaling=NZNSHM_scaling,
                        Nstart=int(Nstart))
            #           shear_wave_fraction=shear_wave_fraction,

                    
    # Prepare waveforms and synthetics       
    if make_GFs==1 or make_synthetics==1:
        runslip.inversionGFs(home,project_name,GF_list,None,fault_name,model_name,
            dt,None,NFFT,None,make_GFs,make_synthetics,dk,pmin,
            pmax,kmax,0,time_epi,hot_start,ncpus,custom_stf,impulse=True) 

    #Make low frequency waveforms
    if make_waveforms==1:
        forward.waveforms_fakequakes(home,project_name,fault_name,rupture_list,GF_list,
                    model_name,run_name,dt,NFFT,G_from_file,G_name,source_time_function,
                    stf_falloff_rate,hot_start=hot_start)

    #Generate static offsets
    if make_statics==1:
        forward.coseismics_fakequakes(home,project_name,GF_list,G_from_file,G_name_static,
                            model_name,rupture_list)

    #Make semistochastic HF waveforms         
    if make_hf_waveforms==1:
        forward.hf_waveforms(home,project_name,fault_name,rupture_list,GF_list,
                    model_name,run_name,dt,NFFT,G_from_file,G_name,rise_time_depths,
                    moho_depth_in_km,ncpus,source_time_function=source_time_function,
                    duration=duration,stf_falloff_rate=stf_falloff_rate,hf_dt=hf_dt,
                    Pwave=Pwave,hot_start=hot_start,stress_parameter=stress_parameter,
                    high_stress_depth=high_stress_depth)

    # Combine LF and HF waveforms with match filter                              
    if match_filter==1:
        forward.match_filter(home,project_name,fault_name,rupture_list,GF_list,
                zero_phase,order,fcorner)
                    
                
