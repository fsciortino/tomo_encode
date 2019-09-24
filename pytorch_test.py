# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:48:02 2016
@author: normandy
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import readline
import MDSplus
import profiletools
import cPickle as pkl
import eqtools



# List of shots for training
cases = {}
cases[1101014006] = {'line': ['Ca/6'], 'tht':[0]}


'''
cases[1101014019] = {'line': ['Ca/6'], 'tht':[0,4]}
cases[1101014029] = {'line': ['Ca/6','Ar/2'], 'tht':[0,1]}
#cases[1101014030] = {'line': ['Ca/6','Ar/2'], 'tht':[0,1]}  #Ar/2 is burnt
cases[1101014030] = {'line': ['Ca/6'], 'tht':[0]}

cases[1120914029] = {'line': ['Ca/9','Ar/7'], 'tht':[9,8]}
cases[1120914036] = {'line': ['Ca/9','Ca/9','Ar/7'], 'tht':[5,6,1]}

cases[1100305019] = {'line': ['Ca/6'], 'tht':[9]}
cases[1140729021] = {'line': ['Ca/6','Ar/2'], 'tht':[9,8]}
cases[1140729023] = {'line': ['Ca/6','Ar/2'], 'tht':[9,8]}
cases[1140729030] = {'line': ['Ca/6','Ar/2'], 'tht':[9,8]}
'''

class ThacoData:
    def __init__(self, node):

        proNode = node.getNode('PRO')
        rhoNode = node.getNode('RHO')
        perrNode = node.getNode('PROERR')

        rpro = proNode.data()
        rrho = rhoNode.data()
        rperr = perrNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]


def unbiased_weighted_var(values, weights, axis):
    cleaned_values = np.ma.masked_array(values, np.isnan(values))
    average = np.ma.average(cleaned_values, axis=axis, weights=weights)

    num = np.sum((values - average)**2*weights, axis=axis)
    denom = np.sum(weights,axis=axis) - np.sum(weights**2,axis=axis)/np.sum(weights,axis=axis)

    return num/denom



# Create training set
for shot,dic in cases.iteritems():
    print shot,": ",dic
    
    t_min = 0.0 #1.2
    t_max = 2.0 #1.3

    Ti_bias_A = 0.0 
    Ti_bias_B = 0.33
    merge_pnt = 0.35

    # measurements in the tree are given on a psi_norm grid
    e = eqtools.CModEFITTree(shot)

    specTree = MDSplus.Tree('spectroscopy', shot)

    nodeA = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS1.HELIKE.PROFILES.Z')
    nodeB = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

    # branch B is on Ca18+, branch A on Ar16+
    dataA = ThacoData(nodeA)
    dataB = ThacoData(nodeB)

    indexA0 = np.searchsorted(dataA.time, t_min)
    indexA1 = np.searchsorted(dataA.time, t_max)
    indexB0 = np.searchsorted(dataB.time, t_min)
    indexB1 = np.searchsorted(dataB.time, t_max)

    # Ti
    Ti_A_vals = dataA.pro[3,indexA0:indexA1,:]
    Ti_B_vals = dataB.pro[3,indexB0:indexB1,:]
    Ti_A_uncs = dataA.perr[3,indexA0:indexA1,:]
    Ti_B_uncs = dataB.perr[3,indexB0:indexB1,:]

    # ensure only positive values are counted -- set others to 0 (will make network robust)
    Ti_A_vals[(Ti_A_vals<0) | (Ti_A_vals>5)] = 0.0
    Ti_B_vals[(Ti_B_vals<0) | (Ti_B_vals>5)] = 0.0

    Ti_A_uncs[(Ti_A_uncs<0) | (Ti_A_vals>5)] = 0.0
    Ti_B_uncs[(Ti_B_uncs<0) | (Ti_B_vals>5)] = 0.0

    '''
    # use masked arrays to ensure that nan's are ignored in averages
    Ti_A_vals_clean = np.ma.masked_array(Ti_A_vals, np.isnan(Ti_A_vals))
    Ti_A_uncs_clean = np.ma.masked_array(Ti_A_uncs, np.isnan(Ti_A_uncs))
    Ti_mean_A = np.ma.average(Ti_A_vals_clean,axis=0, weights = 1.0/(Ti_A_uncs_clean**2))

    Ti_B_vals_clean = np.ma.masked_array(Ti_B_vals, np.isnan(Ti_B_vals))
    Ti_B_uncs_clean = np.ma.masked_array(Ti_B_uncs, np.isnan(Ti_B_uncs))
    Ti_mean_B = np.ma.average(Ti_B_vals_clean, axis=0, weights = 1.0/(Ti_B_uncs_clean**2))
    '''

    # omega_tor
    omega_tor_A_vals = dataA.pro[1,indexA0:indexA1,:]
    omega_tor_B_vals = dataB.pro[1,indexB0:indexB1,:]
    omega_tor_A_uncs = dataA.perr[1,indexA0:indexA1,:]
    omega_tor_B_uncs = dataB.perr[1,indexB0:indexB1,:]

    # assume that all velocities should be of the mean sign
    # if np.sign(np.mean(omega_tor_A_vals))>0:
    #     mean_sign = 0
    omega_tor_A_vals[(omega_tor_A_vals<0) | (omega_tor_A_vals>300)] = 0.0
    omega_tor_B_vals[(omega_tor_B_vals<0) | (omega_tor_B_vals>300)] = 0.0

    omega_tor_A_uncs[(omega_tor_A_uncs<0) | (omega_tor_A_vals>300)] = 0.0
    omega_tor_B_uncs[(omega_tor_B_uncs<0) | (omega_tor_B_vals>300)] = 0.0

    omega_tor_A_vals_clean = np.ma.masked_array(omega_tor_A_vals, np.isnan(omega_tor_A_vals))
    omega_tor_A_uncs_clean = np.ma.masked_array(omega_tor_A_uncs, np.isnan(omega_tor_A_uncs))
    omega_tor_mean_A = np.ma.average(omega_tor_A_vals_clean,axis=0, weights = 1.0/(omega_tor_A_uncs_clean**2))

    omega_tor_B_vals_clean = np.ma.masked_array(omega_tor_B_vals, np.isnan(omega_tor_B_vals))
    omega_tor_B_uncs_clean = np.ma.masked_array(omega_tor_B_uncs, np.isnan(omega_tor_B_uncs))
    omega_tor_mean_B = np.ma.average(omega_tor_B_vals_clean, axis=0, weights = 1.0/(omega_tor_B_uncs_clean**2))



#  =================================
    # Ti
    if Ti_A_vals.shape[0]>1:
        Ti_var_A = unbiased_weighted_var(Ti_A_vals, 1.0/(Ti_A_uncs**2), axis=0)
        Ti_std_A = np.asarray([np.sqrt(val) for val in Ti_var_A])
    else:
        Ti_std_A = Ti_A_uncs[0,:]

    if Ti_B_vals.shape[0]>1:
        Ti_var_B = unbiased_weighted_var(Ti_B_vals, 1.0/(Ti_B_uncs**2), axis=0)
        Ti_std_B = np.asarray([np.sqrt(val) for val in Ti_var_B])
    else:
        Ti_std_B = Ti_B_uncs[0,:]

    # vtor
    if omega_tor_A_vals.shape[0]>1:
        omega_tor_var_A = unbiased_weighted_var(omega_tor_A_vals, 1.0/(omega_tor_A_uncs**2), axis=0)
        omega_tor_std_A = np.asarray([np.sqrt(val) for val in omega_tor_var_A])
    else:
        omega_tor_std_A = omega_tor_A_uncs[0,:]

    if omega_tor_B_vals.shape[0]>1:
        omega_tor_var_B = unbiased_weighted_var(omega_tor_B_vals, 1.0/(omega_tor_B_uncs**2), axis=0)
        omega_tor_std_B = np.asarray([np.sqrt(val) for val in omega_tor_var_B])
    else:
        omega_tor_std_B = omega_tor_B_uncs[0,:]


    plt.figure()
    plt.errorbar(dataA.rho, Ti_mean_A, yerr=Ti_std_A)
    plt.errorbar(dataB.rho, Ti_mean_B-0.33, yerr=Ti_std_B)

    plt.figure()
    plt.errorbar(dataA.rho, omega_tor_mean_A, yerr=omega_tor_std_A)
    plt.errorbar(dataB.rho, omega_tor_mean_B, yerr=omega_tor_std_B)

    # ==========================================
    # merging position
    merge_idx_A=np.argmin(np.abs(dataA.rho-merge_pnt))
    merge_idx_B=np.argmin(np.abs(dataB.rho-merge_pnt))

    # merge
    x_B = dataB.rho[:merge_idx_B]
    x_A = dataA.rho[merge_idx_A:]

    # Ti
    y_B = Ti_mean_B[:merge_idx_B] - Ti_bias_B
    y_A = Ti_mean_A[merge_idx_A:] - Ti_bias_A

    y_err_B = Ti_std_B[:merge_idx_B]
    y_err_A = Ti_std_A[merge_idx_A:] 

    ti_x = np.concatenate((x_B, x_A), axis=0)
    ti_y = np.concatenate((y_B, y_A), axis=0)
    ti_y_err = np.concatenate((y_err_B,y_err_A), axis=0)

    ti_roa = e.rho2rho('psinorm', 'r/a', ti_x, (t_min+t_max)/2.0)

    plt.figure()
    plt.errorbar(ti_roa, ti_y, ti_y_err)
    plt.xlabel('r/a'); plt.ylabel(r'$T_i$ [keV]')

    # omega_tor
    y_B = omega_tor_mean_B[:merge_idx_B] 
    y_A = omega_tor_mean_A[merge_idx_A:] 

    y_err_B = omega_tor_std_B[:merge_idx_B]
    y_err_A = omega_tor_std_A[merge_idx_A:] 

    omega_tor_x = np.concatenate((x_B, x_A), axis=0)
    omega_tor_y = np.concatenate((y_B, y_A), axis=0)
    omega_tor_y_err = np.concatenate((y_err_B,y_err_A), axis=0)

    omega_tor_roa = e.rho2rho('psinorm', 'r/a', omega_tor_x, (t_min+t_max)/2.0)

    plt.figure()
    plt.errorbar(omega_tor_roa, omega_tor_y, omega_tor_y_err)
    plt.xlabel('r/a'); plt.ylabel(r'$\omega_{tor}$ [kHz]')

#  ================================= Save results

ti_res = (ti_roa, ti_y, ti_y_err, (t_min,t_max))
with open('/home/sciortino/fits/tifit_%d_bmix.pkl'%shot,'wb') as f:
    pkl.dump(ti_res, f, protocol=pkl.HIGHEST_PROTOCOL)


omega_tor_res = (omega_tor_roa, omega_tor_y, omega_tor_y_err, (t_min,t_max))
with open('/home/sciortino/fits/omega_tor_fit_%d_bmix.pkl'%shot,'wb') as f:
    pkl.dump(omega_tor_res, f, protocol=pkl.HIGHEST_PROTOCOL)
