## A variety of scripts to help with the naming of files

def get_rupture_df_name(fault_id:str , deficit_mod:str, velmod:str, rupt_lock:bool, NZNSHMscaling: bool, uniformSlip: bool, nrupts=50000, old_format=False):
    """
    Fault ID: Identifier for the folder (e.g., 'hikkerm, 'FQ_')
    deficit_mod: Deficit model used (e.g., 'tenchlock')
    velmod: rigidity model (e.g. '3e10', 'wuatom')
    rupt_lock: True if ruptures generate with locking model
    NSNSHM_scaling: True if scaling to NSHM area
    uniformslip: True if uniform slip is used
    nrupts: Number of ruptures generated (default is 50000)
    
    """

    lock = "_locking" if rupt_lock else "_nolocking"
    NZNSHM = "_NZNSHMscaling" if NZNSHMscaling else ""
    uniform = "_uniformSlip" if uniformSlip else ""

    if old_format:
        rupture_df_file = f'{fault_id}_{velmod}{lock}{NZNSHM}{uniform}_df_n50000.csv'

    else:
        NZNSHM = "_NSHMarea" if NZNSHMscaling else ""
        rupture_df_file = f'{fault_id}_{deficit_mod}_{velmod}{NZNSHM}{uniform}_df_n{nrupts}.csv'

    return rupture_df_file

def get_inv_results_tag(n_ruptures: float, slip_weight: int, GR_weight: int, norm_weight=None, nrupt_weight=None, nrupt_cuttoff=None, taper_max_Mw=None, alpha_s=None, b=None, N=None, pMax=-1, max_iter='*'):
    # Script to generate the tag for the inverted results from deficit inversion

    # Optional tags
    norm_tag = f"_N{norm_weight}" if norm_weight else ""
    nrupt_tag = f"_nr{nrupt_weight}{nrupt_cuttoff}" if nrupt_weight else ""
    pMax_tag = f"_pMax{pMax}" if pMax != -1 else "" ""
    if taper_max_Mw is not None:
        taper_tag = f"_taper{taper_max_Mw}Mw_alphas{alpha_s:.1f}".replace('.', '-')
        n5 = ""
    else:
        taper_tag = ""
        n5 = f"_N{str(N)}".replace('.', '-')
    max_iter_tag = f"_nIt{max_iter}" if max_iter is not None else ""
        

    results_tag = f"n{n_ruptures}_S{slip_weight}{norm_tag}_GR{GR_weight}{nrupt_tag}{taper_tag}_b{str(b).replace('.','-')}{n5}{pMax_tag}{max_iter_tag}"

    return results_tag