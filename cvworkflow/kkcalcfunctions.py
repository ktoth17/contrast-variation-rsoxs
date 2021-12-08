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
        Incident angle in radians.
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

def kkcalc_plot(delta, beta, *, label, min_ev, max_ev):
    min_beta = min(beta[:,1])
    max_beta = max(beta[:,1])
    min_delta = min(delta[:,1])
    max_delta = max(delta[:,1])
    plt.figure()
    plt.plot(delta[:, 0], delta[:, 1], label=label, color = 'r')
    plt.legend()
    plt.xlim(min_ev, max_ev)
    plt.ylim(0.9*min_delta, 1.1*max_delta)
    plt.title('{:d} eV - {:d} eV'.format(min_ev, max_ev),fontsize=16)
    plt.xlabel('Energy [eV]',fontsize=16)
    plt.ylabel(r'$\delta$',fontsize=16)
    plt.show()

    plt.figure()
    plt.plot(beta[:, 0], beta[:, 1], label=label, color = 'b')
    plt.legend()
    plt.xlim(min_ev, max_ev)
    plt.ylim(0.9*min_beta, 1.1*max_beta)
    plt.title('{:d} eV - {:d} eV'.format(min_ev, max_ev),fontsize=16)
    plt.xlabel('Energy [eV]',fontsize=16)
    plt.ylabel(r'$\beta$',fontsize=16)
    plt.show()
