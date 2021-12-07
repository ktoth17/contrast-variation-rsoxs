# cvworkflow/filefunctions.py

import numpy as np
import pandas as pd

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
