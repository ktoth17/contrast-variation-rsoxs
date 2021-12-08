# cvworkflow/filefunctions.py

import pathlib
import os
import re
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.interpolate import griddata
from scipy import stats
from scipy import interpolate

def read_rsoxs_data(dataPath, reg, new_q_intervals=300):
    nfiles = os.listdir(dataPath)
    n_files = len(nfiles)

    #Read in the files in the folder (in .dat format)
    df_list = []
    for file in sorted(dataPath.glob('*dat'),key=lambda x: float(reg.search(str(x)).groups()[0])):  #Used to be 'ISV*dat'
        #energy = float(reg.search(str(file)).groups()[0].replace('p','.'))
        energy = float(reg.search(str(file)).groups()[0])
        sdf_np = np.loadtxt(file) #load in the data from .dat IgorPro files
        sdf_all = pd.DataFrame(sdf_np) #convert to pandas Dataframe
        sdf_all.columns = ['q','I','unk1','unk2'] #all columns

        sdf = sdf_all[['q', 'I']].copy() #new dataframe with only q and I
        sdf = sdf.set_index('q').squeeze()
        sdf.name = energy
        # Double normalization:
        #norm = au_mesh_avg[i]*(c_waxs_diode_avg[i]/c_au_mesh_avg[i])
        #sdf = sdf/norm
        df_list.append(sdf)
    min_q = max([sdf.index.min() for sdf in df_list])
    max_q = min([sdf.index.max() for sdf in df_list])

    new_q = np.geomspace(min_q,max_q,new_q_intervals)

    # Interpolate scattering data onto a common grid
    new_df_list = []
    n_files = len(df_list)
    for sdf in df_list:
        nsdf = (sdf
                .reindex(sdf.index.union(new_q))
                .interpolate(method='linear')
                .reindex(new_q)
               )
        new_df_list.append(nsdf)
    df = pd.concat(new_df_list,axis=1)

    xrf_fit_values=xrf_subtraction(df)

    new_df_list_xrf = []
    i = 0
    for sdf in df_list:
        nsdf = (sdf
                .reindex(sdf.index.union(new_q))
                .interpolate(method='linear')
                .reindex(new_q)
               )
        #Subtracting X-ray fluorescence
        if float(nsdf.name) > 283.9:
            nsdf = nsdf - xrf_fit_values[i]

        # Double normalization:
        #norm = au_mesh_avg[i]*(c_waxs_diode_avg[i]/c_au_mesh_avg[i])
        #nsdf = nsdf/norm
        new_df_list_xrf.append(nsdf)
        i = i+1

    df_xrf = pd.concat(new_df_list_xrf,axis=1)

    return df_xrf

def xrf_subtraction(df, iq_start = 200,iq_end = 300,range_len=55):
    # number 10 is 280 eV, number 55 is 310 eV
    # above 0.1 nm^-1
    q_values = df.iloc[iq_start:iq_end,10].index.to_numpy()

    xrf_fit_values = []
    xrf_fit_A_values = []
    for i in range(range_len):  #use 61 for ISV data!!
        intensity = df.iloc[iq_start:iq_end,i].to_numpy()
        init_guess = [1, 0]
        para_best, ier = scipy.optimize.leastsq(err, init_guess, args=(q_values,intensity))
        xrf_fit_values.append(para_best[1])
        xrf_fit_A_values.append(para_best[0])
    return xrf_fit_values

# Exponential function
def exp_func(q, para):
    A, C = para
    return A * q**(-3.7) + C

def err(para,q,y):
    return abs(exp_func(q, para)-y)

def read_rsoxs_currents(path, scan_id,exposure_time=2,time_avg_width=1, min_ev=270.1, max_ev=330):
    # read primary csv for energies, need to add it outside the scan_id folder
    primary = pd.read_csv(list(path.glob(f'{scan_id}*primary*.csv'))[0])
    energy_df = primary['en_energy'].copy()
    energy = energy_df.to_numpy()

    # read in Au mesh current, Sample Shutter current, and diode current from csv files inside scan_id folder
    folder_path = pathlib.Path(f'./{scan_id}')
    exact_energy = pd.read_csv(list(folder_path.glob(f'{scan_id}*monoen_readback_monitor*.csv'))[0])
    au_mesh = pd.read_csv(list(folder_path.glob(f'{scan_id}*Mesh*.csv'))[0])
    sample = pd.read_csv(list(folder_path.glob(f'{scan_id}*Sample Current*.csv'))[0])
    shutter_toggle = pd.read_csv(list(folder_path.glob(f'{scan_id}*Shutter*.csv'))[0])
    waxs_diode = pd.read_csv(list(folder_path.glob(f'{scan_id}*WAXS Beamstop*.csv'))[0])

    open_close = shutter_toggle['RSoXS Shutter Toggle']*shutter_toggle['time']
    start_time = open_close[open_close!=0]
    exact_energy_avg = np.zeros(len(start_time))
    au_mesh_avg = np.zeros(len(start_time))
    au_mesh_std = np.zeros(len(start_time))
    waxs_diode_avg = np.zeros(len(start_time))
    waxs_diode_std = np.zeros(len(start_time))
    sample_avg = np.zeros(len(start_time))
    sample_std = np.zeros(len(start_time))

    for i,time in enumerate(start_time):
        mid_time_point = time + exposure_time/2
        #avg_idx = (au_mesh['time'] > time) & (au_mesh['time'] < (time + exposure_time)) #originally
        avg_idx = (au_mesh['time'] > (mid_time_point-time_avg_width/2)) & (au_mesh['time'] < (mid_time_point+time_avg_width/2))
        au_mesh_avg[i] = np.mean(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        au_mesh_std[i] = np.std(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        #avg_idx2 = (waxs_diode['time'] > time) & (waxs_diode['time'] < (time + exposure_time)) #originally
        avg_idx2 = (waxs_diode['time'] > (mid_time_point-time_avg_width/2)) & (waxs_diode['time'] < (mid_time_point+time_avg_width/2))
        waxs_diode_avg[i] = np.mean(waxs_diode['WAXS Beamstop'][avg_idx2])
        waxs_diode_std[i] = np.std(waxs_diode['WAXS Beamstop'][avg_idx2])
        #avg_idx3 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx3 = (sample['time'] > (mid_time_point-time_avg_width/2)) & (sample['time'] < (mid_time_point+time_avg_width/2))
        sample_avg[i] = np.mean(sample['RSoXS Sample Current'][avg_idx3])
        sample_std[i] = np.std(sample['RSoXS Sample Current'][avg_idx3])
        #avg_idx4 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx4 = (exact_energy['time'] > (mid_time_point-time_avg_width/2)) & (exact_energy['time'] < (mid_time_point+time_avg_width/2))
        exact_energy_avg[i] = np.mean(exact_energy['en_monoen_readback'][avg_idx4])


    # Interpolate all currents to common energy grid of 100 values between 270 and 355 (high energy cut off since 340 occurs double)
    energy_new = np.geomspace(min_ev,max_ev,1000)
    f1 = interpolate.interp1d(exact_energy_avg, au_mesh_avg, kind='linear')
    au_mesh_avg_new = f1(energy_new)

    f2 = interpolate.interp1d(exact_energy_avg, au_mesh_std, kind='linear')
    au_mesh_std_new = f2(energy_new)

    f3 = interpolate.interp1d(exact_energy_avg, waxs_diode_avg, kind='linear')
    waxs_diode_avg_new = f3(energy_new)

    f4 = interpolate.interp1d(exact_energy_avg, waxs_diode_std, kind='linear')
    waxs_diode_std_new = f4(energy_new)

    f5 = interpolate.interp1d(exact_energy_avg, sample_avg, kind='linear')
    sample_avg_new = f5(energy_new)

    f6 = interpolate.interp1d(exact_energy_avg, sample_std, kind='linear')
    sample_std_new = f6(energy_new)


    # Weird bug in SST-1 scan code where the last energy is repeated, i.e. 340eV, 339.99eV
    #if (abs(energy[-2] - energy[-1])) < 0.1:
    #    energy = energy[:-1]
    #    au_mesh_avg = au_mesh_avg[:-1]
    #    au_mesh_std = au_mesh_std[:-1]
    #    waxs_diode_avg = waxs_diode_avg[:-1]
    #    waxs_diode_std = waxs_diode_std[:-1]
    #    sample_avg = sample_avg[:-1]
    #    sample_std = sample_std[:-1]
    #    pass

    return energy_new, au_mesh_avg_new, au_mesh_std_new, waxs_diode_avg_new, waxs_diode_std_new, sample_avg_new, sample_std_new

def read_rsoxs_currents_short(path, scan_id,exposure_time=2,time_avg_width=1, min_ev=270.1, max_ev=330):
    # read primary csv for energies, need to add it outside the scan_id folder
    primary = pd.read_csv(list(path.glob(f'{scan_id}*primary*.csv'))[0])
    energy_df = primary['en_energy'].copy()
    energy = energy_df.to_numpy()

    # read in Au mesh current, Sample Shutter current, and diode current from csv files inside scan_id folder
    folder_path = pathlib.Path(f'./{scan_id}')
    exact_energy = pd.read_csv(list(folder_path.glob(f'{scan_id}*monoen_readback_monitor*.csv'))[0])
    au_mesh = pd.read_csv(list(folder_path.glob(f'{scan_id}*Mesh*.csv'))[0])
    sample = pd.read_csv(list(folder_path.glob(f'{scan_id}*Sample Current*.csv'))[0])
    shutter_toggle = pd.read_csv(list(folder_path.glob(f'{scan_id}*Shutter*.csv'))[0])
    waxs_diode = pd.read_csv(list(folder_path.glob(f'{scan_id}*WAXS Beamstop*.csv'))[0])

    open_close = shutter_toggle['RSoXS Shutter Toggle']*shutter_toggle['time']
    start_time = open_close[open_close!=0]
    exact_energy_avg = np.zeros(len(start_time))
    au_mesh_avg = np.zeros(len(start_time))
    au_mesh_std = np.zeros(len(start_time))
    waxs_diode_avg = np.zeros(len(start_time))
    waxs_diode_std = np.zeros(len(start_time))
    sample_avg = np.zeros(len(start_time))
    sample_std = np.zeros(len(start_time))

    for i,time in enumerate(start_time):
        mid_time_point = time + exposure_time/2
        #avg_idx = (au_mesh['time'] > time) & (au_mesh['time'] < (time + exposure_time)) #originally
        avg_idx = (au_mesh['time'] > (mid_time_point-time_avg_width/2)) & (au_mesh['time'] < (mid_time_point+time_avg_width/2))
        au_mesh_avg[i] = np.mean(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        au_mesh_std[i] = np.std(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        #avg_idx2 = (waxs_diode['time'] > time) & (waxs_diode['time'] < (time + exposure_time)) #originally
        avg_idx2 = (waxs_diode['time'] > (mid_time_point-time_avg_width/2)) & (waxs_diode['time'] < (mid_time_point+time_avg_width/2))
        waxs_diode_avg[i] = np.mean(waxs_diode['WAXS Beamstop'][avg_idx2])
        waxs_diode_std[i] = np.std(waxs_diode['WAXS Beamstop'][avg_idx2])
        #avg_idx3 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx3 = (sample['time'] > (mid_time_point-time_avg_width/2)) & (sample['time'] < (mid_time_point+time_avg_width/2))
        sample_avg[i] = np.mean(sample['RSoXS Sample Current'][avg_idx3])
        sample_std[i] = np.std(sample['RSoXS Sample Current'][avg_idx3])
        #avg_idx4 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx4 = (exact_energy['time'] > (mid_time_point-time_avg_width/2)) & (exact_energy['time'] < (mid_time_point+time_avg_width/2))
        exact_energy_avg[i] = np.mean(exact_energy['en_monoen_readback'][avg_idx4])


    # Interpolate all currents to common energy grid of 100 values between 270 and 355 (high energy cut off since 340 occurs double)
    energy_new = ([270.1,  272.0,  274.0,  276.0, 278.0,  280.0, 282.0, 282.25,  282.5, 282.75,283.0, 283.25,  283.5, 283.75, 284.0,284.25,  284.5, 284.75,285.0,285.25,  285.5, 285.75,  286.0,  286.5,  287.0,  287.5,288.0,  288.5,  289.0,  289.5,  290.0,  290.5,  291.0,  291.5,292.0,  293.0,  294.0,  295.0,  296.0,  297.0,  298.0,  299.0,300.0,  301.0,  302.0,  303.0,  304.0,  305.0,  306.0,  310.0,314.0,  318.0,  320.0,  330.0,  340.0])
    f1 = interpolate.interp1d(exact_energy_avg, au_mesh_avg, kind='linear')
    au_mesh_avg_new = f1(energy_new)

    f2 = interpolate.interp1d(exact_energy_avg, au_mesh_std, kind='linear')
    au_mesh_std_new = f2(energy_new)

    f3 = interpolate.interp1d(exact_energy_avg, waxs_diode_avg, kind='linear')
    waxs_diode_avg_new = f3(energy_new)

    f4 = interpolate.interp1d(exact_energy_avg, waxs_diode_std, kind='linear')
    waxs_diode_std_new = f4(energy_new)

    f5 = interpolate.interp1d(exact_energy_avg, sample_avg, kind='linear')
    sample_avg_new = f5(energy_new)

    f6 = interpolate.interp1d(exact_energy_avg, sample_std, kind='linear')
    sample_std_new = f6(energy_new)


    # Weird bug in SST-1 scan code where the last energy is repeated, i.e. 340eV, 339.99eV
    #if (abs(energy[-2] - energy[-1])) < 0.1:
    #    energy = energy[:-1]
    #    au_mesh_avg = au_mesh_avg[:-1]
    #    au_mesh_std = au_mesh_std[:-1]
    #    waxs_diode_avg = waxs_diode_avg[:-1]
    #    waxs_diode_std = waxs_diode_std[:-1]
    #    sample_avg = sample_avg[:-1]
    #    sample_std = sample_std[:-1]
    #    pass

    # returns all numpy arrays
    #return energy, au_mesh_avg, au_mesh_std, waxs_diode_avg, waxs_diode_std, sample_avg, sample_std
    return energy_new, au_mesh_avg_new, au_mesh_std_new, waxs_diode_avg_new, waxs_diode_std_new, sample_avg_new, sample_std_new

def read_nexafs_currents(path, scan_id,exposure_time=2,time_avg_width=1, min_ev=270.1, max_ev=330):
    # read primary csv for energies, need to add it outside the scan_id folder
    primary = pd.read_csv(list(path.glob(f'{scan_id}*primary*.csv'))[0])
    energy_df = primary['en_monoen_readback'].copy()
    energy = energy_df.to_numpy()
    primary_time_df = primary['time'].copy()
    primary_time = primary_time_df.to_numpy()

    # read in Au mesh current, Sample Shutter current, and diode current from csv files inside scan_id folder
    folder_path = pathlib.Path(f'./{scan_id}')
    exact_energy = pd.read_csv(list(folder_path.glob(f'{scan_id}*monoen_readback_monitor*.csv'))[0])
    au_mesh = pd.read_csv(list(folder_path.glob(f'{scan_id}*Mesh*.csv'))[0])
    sample = pd.read_csv(list(folder_path.glob(f'{scan_id}*Sample Current*.csv'))[0])
    waxs_diode = pd.read_csv(list(folder_path.glob(f'{scan_id}*WAXS Beamstop*.csv'))[0])

    exact_energy_avg = np.zeros(len(primary_time))
    au_mesh_avg = np.zeros(len(primary_time))
    au_mesh_std = np.zeros(len(primary_time))
    waxs_diode_avg = np.zeros(len(primary_time))
    waxs_diode_std = np.zeros(len(primary_time))
    sample_avg = np.zeros(len(primary_time))
    sample_std = np.zeros(len(primary_time))

    for i,time in enumerate(primary_time):
        mid_time_point = time
        #mid_time_point = time + exposure_time/2 #Alternative, choosing midpoint furhter along in time
        #avg_idx = (au_mesh['time'] > time) & (au_mesh['time'] < (time + exposure_time)) #originally
        avg_idx = (au_mesh['time'] > (mid_time_point-time_avg_width/2)) & (au_mesh['time'] < (mid_time_point+time_avg_width/2))
        au_mesh_avg[i] = np.mean(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        au_mesh_std[i] = np.std(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        #avg_idx2 = (waxs_diode['time'] > time) & (waxs_diode['time'] < (time + exposure_time)) #originally
        avg_idx2 = (waxs_diode['time'] > (mid_time_point-time_avg_width/2)) & (waxs_diode['time'] < (mid_time_point+time_avg_width/2))
        waxs_diode_avg[i] = np.mean(waxs_diode['WAXS Beamstop'][avg_idx2])
        waxs_diode_std[i] = np.std(waxs_diode['WAXS Beamstop'][avg_idx2])
        #avg_idx3 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx3 = (sample['time'] > (mid_time_point-time_avg_width/2)) & (sample['time'] < (mid_time_point+time_avg_width/2))
        sample_avg[i] = np.mean(sample['RSoXS Sample Current'][avg_idx3])
        sample_std[i] = np.std(sample['RSoXS Sample Current'][avg_idx3])
        avg_idx4 = (exact_energy['time'] > (mid_time_point-time_avg_width/2)) & (exact_energy['time'] < (mid_time_point+time_avg_width/2))
        exact_energy_avg[i] = np.mean(exact_energy['en_monoen_readback'][avg_idx4])


    # Interpolate all currents to common energy grid of 100 values between 270 and 355 (high energy cut off since 340 occurs double)
    energy_new = np.geomspace(min_ev, max_ev,1000)
    f1 = interpolate.interp1d(exact_energy_avg, au_mesh_avg, kind='linear')
    au_mesh_avg_new = f1(energy_new)

    f2 = interpolate.interp1d(exact_energy_avg, au_mesh_std, kind='linear')
    au_mesh_std_new = f2(energy_new)

    f3 = interpolate.interp1d(exact_energy_avg, waxs_diode_avg, kind='linear')
    waxs_diode_avg_new = f3(energy_new)

    f4 = interpolate.interp1d(exact_energy_avg, waxs_diode_std, kind='linear')
    waxs_diode_std_new = f4(energy_new)

    f5 = interpolate.interp1d(exact_energy_avg, sample_avg, kind='linear')
    sample_avg_new = f5(energy_new)

    f6 = interpolate.interp1d(exact_energy_avg, sample_std, kind='linear')
    sample_std_new = f6(energy_new)

    # Weird bug in SST-1 scan code where the last energy is repeated, i.e. 340eV, 339.99eV
    #if (abs(energy[-2] - energy[-1])) < 0.2:
    #    energy = energy[:-1]
    #    au_mesh_avg = au_mesh_avg[:-1]
    #    au_mesh_std = au_mesh_std[:-1]
    #    waxs_diode_avg = waxs_diode_avg[:-1]
    #    waxs_diode_std = waxs_diode_std[:-1]
    #    sample_avg = sample_avg[:-1]
    #    sample_std = sample_std[:-1]
    #    pass

    # returns all numpy arrays
    return energy_new, au_mesh_avg_new, au_mesh_std_new, waxs_diode_avg_new, waxs_diode_std_new, sample_avg_new, sample_std_new

#functions from Peter D.
path = pathlib.Path('./')

#For NEXAFS, shutter is always open!! Need to extract time range from primary file

def read_nexafs_currents_short(path, scan_id,exposure_time=2,time_avg_width=1, min_ev=270.1, max_ev=330):
    # read primary csv for energies, need to add it outside the scan_id folder
    primary = pd.read_csv(list(path.glob(f'{scan_id}*primary*.csv'))[0])
    energy_df = primary['en_monoen_readback'].copy()
    energy = energy_df.to_numpy()
    primary_time_df = primary['time'].copy()
    primary_time = primary_time_df.to_numpy()

    # read in Au mesh current, Sample Shutter current, and diode current from csv files inside scan_id folder
    folder_path = pathlib.Path(f'./{scan_id}')
    exact_energy = pd.read_csv(list(folder_path.glob(f'{scan_id}*monoen_readback_monitor*.csv'))[0])
    au_mesh = pd.read_csv(list(folder_path.glob(f'{scan_id}*Mesh*.csv'))[0])
    sample = pd.read_csv(list(folder_path.glob(f'{scan_id}*Sample Current*.csv'))[0])
    waxs_diode = pd.read_csv(list(folder_path.glob(f'{scan_id}*WAXS Beamstop*.csv'))[0])

    exact_energy_avg = np.zeros(len(primary_time))
    au_mesh_avg = np.zeros(len(primary_time))
    au_mesh_std = np.zeros(len(primary_time))
    waxs_diode_avg = np.zeros(len(primary_time))
    waxs_diode_std = np.zeros(len(primary_time))
    sample_avg = np.zeros(len(primary_time))
    sample_std = np.zeros(len(primary_time))

    for i,time in enumerate(primary_time):
        mid_time_point = time
        #mid_time_point = time + exposure_time/2 #Alternative, choosing midpoint furhter along in time
        #avg_idx = (au_mesh['time'] > time) & (au_mesh['time'] < (time + exposure_time)) #originally
        avg_idx = (au_mesh['time'] > (mid_time_point-time_avg_width/2)) & (au_mesh['time'] < (mid_time_point+time_avg_width/2))
        au_mesh_avg[i] = np.mean(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        au_mesh_std[i] = np.std(au_mesh['RSoXS Au Mesh Current'][avg_idx])
        #avg_idx2 = (waxs_diode['time'] > time) & (waxs_diode['time'] < (time + exposure_time)) #originally
        avg_idx2 = (waxs_diode['time'] > (mid_time_point-time_avg_width/2)) & (waxs_diode['time'] < (mid_time_point+time_avg_width/2))
        waxs_diode_avg[i] = np.mean(waxs_diode['WAXS Beamstop'][avg_idx2])
        waxs_diode_std[i] = np.std(waxs_diode['WAXS Beamstop'][avg_idx2])
        #avg_idx3 = (sample['time'] > time) & (sample['time'] < (time + exposure_time)) #originally
        avg_idx3 = (sample['time'] > (mid_time_point-time_avg_width/2)) & (sample['time'] < (mid_time_point+time_avg_width/2))
        sample_avg[i] = np.mean(sample['RSoXS Sample Current'][avg_idx3])
        sample_std[i] = np.std(sample['RSoXS Sample Current'][avg_idx3])
        avg_idx4 = (exact_energy['time'] > (mid_time_point-time_avg_width/2)) & (exact_energy['time'] < (mid_time_point+time_avg_width/2))
        exact_energy_avg[i] = np.mean(exact_energy['en_monoen_readback'][avg_idx4])


    # Interpolate all currents to common energy grid of 100 values between 270 and 355 (high energy cut off since 340 occurs double)
    energy_new = ([270.1,  272.0,  274.0,  276.0, 278.0,  280.0, 282.0, 282.25,  282.5, 282.75,283.0, 283.25,  283.5, 283.75, 284.0,284.25,  284.5, 284.75,285.0,285.25,  285.5, 285.75,  286.0,  286.5,  287.0,  287.5,288.0,  288.5,  289.0,  289.5,  290.0,  290.5,  291.0,  291.5,292.0,  293.0,  294.0,  295.0,  296.0,  297.0,  298.0,  299.0,300.0,  301.0,  302.0,  303.0,  304.0,  305.0,  306.0,  310.0,314.0,  318.0,  320.0,  330.0,  340.0])
    f1 = interpolate.interp1d(exact_energy_avg, au_mesh_avg, kind='linear')
    au_mesh_avg_new = f1(energy_new)

    f2 = interpolate.interp1d(exact_energy_avg, au_mesh_std, kind='linear')
    au_mesh_std_new = f2(energy_new)

    f3 = interpolate.interp1d(exact_energy_avg, waxs_diode_avg, kind='linear')
    waxs_diode_avg_new = f3(energy_new)

    f4 = interpolate.interp1d(exact_energy_avg, waxs_diode_std, kind='linear')
    waxs_diode_std_new = f4(energy_new)

    f5 = interpolate.interp1d(exact_energy_avg, sample_avg, kind='linear')
    sample_avg_new = f5(energy_new)

    f6 = interpolate.interp1d(exact_energy_avg, sample_std, kind='linear')
    sample_std_new = f6(energy_new)

    # Weird bug in SST-1 scan code where the last energy is repeated, i.e. 340eV, 339.99eV
    #if (abs(energy[-2] - energy[-1])) < 0.2:
    #    energy = energy[:-1]
    #    au_mesh_avg = au_mesh_avg[:-1]
    #    au_mesh_std = au_mesh_std[:-1]
    #    waxs_diode_avg = waxs_diode_avg[:-1]
    #    waxs_diode_std = waxs_diode_std[:-1]
    #    sample_avg = sample_avg[:-1]
    #    sample_std = sample_std[:-1]
    #    pass

    # returns all numpy arrays
    return energy_new, au_mesh_avg_new, au_mesh_std_new, waxs_diode_avg_new, waxs_diode_std_new, sample_avg_new, sample_std_new
