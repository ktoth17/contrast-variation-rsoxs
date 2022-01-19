# cvworkflow/kkcalcfunctions.py

import kkcalc
from kkcalc import data
from kkcalc import kk

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def kkcalc_convert(file_path, *, chemical_formula, density, min_ev, max_ev, load_options=None,input_data_type='Beta',add_background=False,fix_distortions=False,curve_tolerance=None,curve_recursion=50):
    """
    This file is part of the Kramers-Kronig Calculator software package.
    Copyright (c) 2013 Benjamin Watts, Daniel J. Lauk
    The software is licensed under the terms of the zlib/libpng license.
    For details see LICENSE.txt

    Benjamin Watts "Calculation of the Kramers-Kronig transform of X-ray spectra by a piecewise Laurent polynomial method" Opt. Express 22, (2014) 23628-23639. DOI:10.1364/OE.22.023628

    Pip page for kkcalc: https://pypi.org/project/kkcalc/

    Benjamin Watts github: https://github.com/benajamin/kkcalc

    Cloned repository from Daniel Schick with an easy to follow along example: https://github.com/dschick/kkcalc

    Parameters
    ----------
    file_path : pathlib.WindowsPath
        File path to NEXAFS beta spreadsheet, csv or txt. 2 columns: energy and intensity.
    chemical_formula : string
        The chemical formula of the component, i.e. 'C8H8' for polystyrene.
    density : float
        Density of the component in grams/cc. Typically around 1 grams/cc.

    Returns
    -------
    delta : numpy.ndarray
        Real (dispersive) components of the complex index of refraction. Two columns, energy and delta values.
    beta : numpy.ndarray
        Imaginary (absorptive) components of the complex index of refraction. Two columns, energy and beta values.

    Examples
    --------
    Calculate the complex index of refraction of polystrene (PS) from the NEXAFS of PS given from a txt file.

    >>> kkcalc_convert(file_path, *, chemical_formula= 'C8H8', density = 1.05, min_ev=270, max_ev=325)

    """

    merge_points=[min_ev, max_ev]

    output = kk.kk_calculate_real(file_path,
                              chemical_formula,
                              load_options,
                              input_data_type,
                              merge_points,
                              add_background,
                              fix_distortions,
                              curve_tolerance,
                              curve_recursion)

    stoichiometry = kk.data.ParseChemicalFormula(chemical_formula)
    formula_mass = data.calculate_FormulaMass(stoichiometry)
    ASF_E, ASF_Data = kk.data.calculate_asf(stoichiometry)
    ASF_Data2 = kk.data.coeffs_to_ASF(ASF_E, np.vstack((ASF_Data, ASF_Data[-1])))
    delta = data.convert_data(output[:,[0,1]],'ASF','refractive_index', Density=density, Formula_Mass=formula_mass)
    beta = data.convert_data(output[:,[0,2]],'ASF','refractive_index', Density=density, Formula_Mass=formula_mass)

    return delta, beta

def kkcalc_plot(delta, beta, *, label, min_ev, max_ev, delta_ylim=[-0.006,0.004], beta_ylim=[0,0.008]):
    plt.figure()
    plt.plot(delta[:, 0], delta[:, 1], label=label, color = 'r')
    plt.legend()
    plt.xlim(min_ev, max_ev)
    plt.ylim(delta_ylim)
    plt.title('{:d} eV - {:d} eV'.format(min_ev, max_ev),fontsize=16)
    plt.xlabel('Energy [eV]',fontsize=16)
    plt.ylabel(r'$\delta$',fontsize=16)
    plt.show()

    plt.figure()
    plt.plot(beta[:, 0], beta[:, 1], label=label, color = 'b')
    plt.legend()
    plt.xlim(min_ev, max_ev)
    plt.ylim(beta_ylim)
    plt.title('{:d} eV - {:d} eV'.format(min_ev, max_ev),fontsize=16)
    plt.xlabel('Energy [eV]',fontsize=16)
    plt.ylabel(r'$\beta$',fontsize=16)
    plt.show()

def component_df(delta, beta, new_q_index, label):
    delta_df = pd.DataFrame(delta[:, 1], columns=['delta_'+label])
    delta_df = delta_df.set_axis(delta[:, 0], axis=0)
    delta_df_new = (delta_df.reindex(delta_df.index.union(new_q_index)).interpolate(method='linear').reindex(new_q_index))

    beta_df = pd.DataFrame(beta[:, 1], columns=['beta_'+label])
    beta_df = beta_df.set_axis(beta[:, 0], axis=0)
    beta_df_new = (beta_df.reindex(beta_df.index.union(new_q_index)).interpolate(method='linear').reindex(new_q_index))

    return delta_df_new, beta_df_new

def make_contrast_M_3(delta1, beta1, label1, delta2, beta2, label2, delta3, beta3, label3, new_q_index):
    delta_label1 = 'delta_'+label1
    delta_label2 = 'delta_'+label2
    delta_label3 = 'delta_'+label3
    beta_label1 = 'beta_'+label1
    beta_label2 = 'beta_'+label2
    beta_label3 = 'beta_'+label3

    delta1_df, beta1_df = component_df(delta1, beta1, new_q_index, label1)
    delta2_df, beta2_df = component_df(delta2, beta2, new_q_index, label2)
    delta3_df, beta3_df = component_df(delta3, beta3, new_q_index, label3)

    index_df = pd.DataFrame(delta1_df, columns=[delta_label1])
    index_df.insert(1, delta_label2, delta2_df, True)
    index_df.insert(2, delta_label3, delta3_df, True)
    index_df.insert(3, beta_label1, beta1_df, True)
    index_df.insert(4, beta_label2, beta2_df, True)
    index_df.insert(5, beta_label3, beta3_df, True)

    contrast_df = index_df.copy(deep=True)
    contrast_df.columns =['S11','S22','S33','S12','S13','S23']
    energy_fourth_term = contrast_df.index.values**4

    # Self term: delta_i^2 + beta_i^2
    contrast_df['S11']=(index_df[delta_label1]*index_df[delta_label1]+index_df[beta_label1]*index_df[beta_label1])*(energy_fourth_term)
    contrast_df['S22']=(index_df[delta_label2]*index_df[delta_label2]+index_df[beta_label2]*index_df[beta_label2])*(energy_fourth_term)
    contrast_df['S33']=(index_df[delta_label3]*index_df[delta_label3]+index_df[beta_label3]*index_df[beta_label3])*(energy_fourth_term)

    # Cross term: 2(delta_i*delta_j + beta_i*beta_j)
    contrast_df['S12']=2*(index_df[delta_label1]*index_df[delta_label2]+index_df[beta_label1]*index_df[beta_label2])*(energy_fourth_term)
    contrast_df['S13']=2*(index_df[delta_label1]*index_df[delta_label3]+index_df[beta_label1]*index_df[beta_label3])*(energy_fourth_term)
    contrast_df['S23']=2*(index_df[delta_label2]*index_df[delta_label3]+index_df[beta_label2]*index_df[beta_label3])*(energy_fourth_term)

    #Make transfer matrix M
    M = []
    for energy,contrasts in contrast_df.iterrows():
        row = []
        row.append(contrasts['S11'])
        row.append(contrasts['S22'])
        row.append(contrasts['S33'])
        row.append(contrasts['S12'])
        row.append(contrasts['S13'])
        row.append(contrasts['S23'])
        M.append(row)
    M = np.array(M)

    #Make negative values 0 in scattering
    M[M < 0] = 0
    #Transpose to get columns of S11, S22, S33, S12, S13, S23
    M = np.transpose(M)
    return M

def make_contrast_M_3_v2(delta1, beta1, label1, delta2, beta2, label2, delta3, beta3, label3, new_q_index):
    delta_label1 = 'delta_'+label1
    delta_label2 = 'delta_'+label2
    delta_label3 = 'delta_'+label3
    beta_label1 = 'beta_'+label1
    beta_label2 = 'beta_'+label2
    beta_label3 = 'beta_'+label3

    delta1_df, beta1_df = component_df(delta1, beta1, new_q_index, label1)
    delta2_df, beta2_df = component_df(delta2, beta2, new_q_index, label2)
    delta3_df, beta3_df = component_df(delta3, beta3, new_q_index, label3)

    index_df = pd.DataFrame(delta1_df, columns=[delta_label1])
    index_df.insert(1, delta_label2, delta2_df, True)
    index_df.insert(2, delta_label3, delta3_df, True)
    index_df.insert(3, beta_label1, beta1_df, True)
    index_df.insert(4, beta_label2, beta2_df, True)
    index_df.insert(5, beta_label3, beta3_df, True)

    contrast_df = index_df.copy(deep=True)
    contrast_df.columns =['S11','S22','S33','S12','S13','S23']
    energy_fourth_term = contrast_df.index.values**4

    # Self term: delta_i^2 + beta_i^2
    contrast_df['S11']=((1-index_df[delta_label1])*(1-index_df[delta_label1])+index_df[beta_label1]*index_df[beta_label1])*(energy_fourth_term)
    contrast_df['S22']=((1-index_df[delta_label2])*(1-index_df[delta_label2])+index_df[beta_label2]*index_df[beta_label2])*(energy_fourth_term)
    contrast_df['S33']=((1-index_df[delta_label3])*(1-index_df[delta_label3])+index_df[beta_label3]*index_df[beta_label3])*(energy_fourth_term)

    # Cross term: 2(delta_i*delta_j + beta_i*beta_j)
    contrast_df['S12']=4*((1-index_df[delta_label1])*(1-index_df[delta_label2])+index_df[beta_label1]*index_df[beta_label2])*(energy_fourth_term)
    contrast_df['S13']=4*((1-index_df[delta_label1])*(1-index_df[delta_label3])+index_df[beta_label1]*index_df[beta_label3])*(energy_fourth_term)
    contrast_df['S23']=4*((1-index_df[delta_label2])*(1-index_df[delta_label3])+index_df[beta_label2]*index_df[beta_label3])*(energy_fourth_term)

    #Make transfer matrix M
    M = []
    for energy,contrasts in contrast_df.iterrows():
        row = []
        row.append(contrasts['S11'])
        row.append(contrasts['S22'])
        row.append(contrasts['S33'])
        row.append(contrasts['S12'])
        row.append(contrasts['S13'])
        row.append(contrasts['S23'])
        M.append(row)
    M = np.array(M)

    #Make negative values 0 in scattering
    M[M < 0] = 0
    #Transpose to get columns of S11, S22, S33, S12, S13, S23
    M = np.transpose(M)
    return M

def make_contrast_M_2(delta1, beta1, label1, delta2, beta2, label2, new_q_index):
    delta_label1 = 'delta_'+label1
    delta_label2 = 'delta_'+label2
    beta_label1 = 'beta_'+label1
    beta_label2 = 'beta_'+label2

    delta1_df, beta1_df = component_df(delta1, beta1, new_q_index, label1)
    delta2_df, beta2_df = component_df(delta2, beta2, new_q_index, label2)

    index_df = pd.DataFrame(delta1_df, columns=[delta_label1])
    index_df.insert(1, delta_label2, delta2_df, True)
    index_df.insert(2, beta_label1, beta1_df, True)
    index_df.insert(3, beta_label2, beta2_df, True)

    contrast_df = index_df[[delta_label1, delta_label2, beta_label1]].copy(deep=True)
    contrast_df.columns =['S11','S22','S12']
    energy_fourth_term = contrast_df.index.values**4

    # Self term: delta_i^2 + beta_i^2
    contrast_df['S11']=(index_df[delta_label1]*index_df[delta_label1]+index_df[beta_label1]*index_df[beta_label1])*(energy_fourth_term)
    contrast_df['S22']=(index_df[delta_label2]*index_df[delta_label2]+index_df[beta_label2]*index_df[beta_label2])*(energy_fourth_term)

    # Cross term: 2(delta_i*delta_j + beta_i*beta_j)
    contrast_df['S12']=2*(index_df[delta_label1]*index_df[delta_label2]+index_df[beta_label1]*index_df[beta_label2])*(energy_fourth_term)

    #Make transfer matrix M
    M = []
    for energy,contrasts in contrast_df.iterrows():
        row = []
        row.append(contrasts['S11'])
        row.append(contrasts['S22'])
        row.append(contrasts['S12'])
        M.append(row)
    M = np.array(M)

    #Make negative values 0 in scattering
    M[M < 0] = 0
    #Transpose to get columns of S11, S22, S33, S12, S13, S23
    M = np.transpose(M)
    return M

def make_contrast_M_2_v2(delta1, beta1, label1, delta2, beta2, label2, new_q_index):
    delta_label1 = 'delta_'+label1
    delta_label2 = 'delta_'+label2
    beta_label1 = 'beta_'+label1
    beta_label2 = 'beta_'+label2

    delta1_df, beta1_df = component_df(delta1, beta1, new_q_index, label1)
    delta2_df, beta2_df = component_df(delta2, beta2, new_q_index, label2)

    index_df = pd.DataFrame(delta1_df, columns=[delta_label1])
    index_df.insert(1, delta_label2, delta2_df, True)
    index_df.insert(2, beta_label1, beta1_df, True)
    index_df.insert(3, beta_label2, beta2_df, True)

    contrast_df = index_df[[delta_label1, delta_label2, beta_label1]].copy(deep=True)
    contrast_df.columns =['S11','S22','S12']
    energy_fourth_term = contrast_df.index.values**4

    # Self term: delta_i^2 + beta_i^2
    contrast_df['S11']=((1-index_df[delta_label1])*(1-index_df[delta_label1])+index_df[beta_label1]*index_df[beta_label1])*(energy_fourth_term)
    contrast_df['S22']=((1-index_df[delta_label2])*(1-index_df[delta_label2])+index_df[beta_label2]*index_df[beta_label2])*(energy_fourth_term)

    # Cross term: 2(delta_i*delta_j + beta_i*beta_j)
    contrast_df['S12']=4*((1-index_df[delta_label1])*(1-index_df[delta_label2])+index_df[beta_label1]*index_df[beta_label2])*(energy_fourth_term)


    #Make transfer matrix M
    M = []
    for energy,contrasts in contrast_df.iterrows():
        row = []
        row.append(contrasts['S11'])
        row.append(contrasts['S22'])
        row.append(contrasts['S12'])
        M.append(row)
    M = np.array(M)

    #Make negative values 0 in scattering
    M[M < 0] = 0
    #Transpose to get columns of S11, S22, S33, S12, S13, S23
    M = np.transpose(M)
    return M
