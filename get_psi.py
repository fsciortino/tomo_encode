
import numpy as np
import eqtools
import matplotlib.pyplot as plt
import sys
import MDSplus
import eqtools

import scipy
import matplotlib.gridspec as mplgs

plt.ion()



def get_psi_map(shot=1101014006, t_val=1.2):
    ''' Get Psi map only in confined region '''

    e = eqtools.CModEFITTree(shot)

    t = e.getTimeBase()
    t_idx = np.argmin(np.abs(t - t_val))

    # get major radius on axis 
    R0 = e.getMagR()

    # flux on R,Z grids
    psiRZ = e.getFluxGrid()
    rGrid = e.getRGrid(length_unit='m')
    zGrid = e.getZGrid(length_unit='m')

    # coordinates along LCFS
    RLCFS = e.getRLCFS(length_unit='m')
    ZLCFS = e.getZLCFS(length_unit='m')

    fluxPlot = plt.figure(figsize=(6,11))
    gs = mplgs.GridSpec(2,1,height_ratios=[30,1])
    psi = fluxPlot.add_subplot(gs[0,0])
    xlim = psi.get_xlim()
    ylim = psi.get_ylim()

    # dummy plot to get x,ylims
    psi.contour(rGrid,zGrid,psiRZ[0],1)

    # plot LCFS
    maskarr = scipy.where(scipy.logical_or(RLCFS[t_idx] > 0.0,scipy.isnan(RLCFS[t_idx])))
    RLCFSframe = RLCFS[t_idx,maskarr[0]]
    ZLCFSframe = ZLCFS[t_idx,maskarr[0]]
    psi.plot(RLCFSframe,ZLCFSframe,'r',linewidth=2.0,zorder=3)


    # find center (largest flux)
    R0 = e.getMagR()
    Z0 = e.getMagZ()
    psiRZ_masked = np.zeros_like(psiRZ)

    # interpolate psiRZ on denser grid
    Rdense = np.linspace(np.min(rGrid), np.max(rGrid),200)
    Zdense = np.linspace(np.min(zGrid), np.max(zGrid),250)


    RR,ZZ = np.meshgrid(Rdense,Zdense)
    psinormRZ_dense = e.rz2phinorm(RR,ZZ,t,sqrt=True)  #normalized sqrt of poloidal flux coordinate


    # mask out regions at the top and bottom
    Zext = np.tile(ZZ[:,:,None],psinormRZ_dense.shape[0])
    cond1 = np.rollaxis(np.tile(ZZ[:,:,None],psinormRZ_dense.shape[0]),axis=2) < np.min(ZLCFSframe)
    cond2 = np.rollaxis(np.tile(ZZ[:,:,None],psinormRZ_dense.shape[0]),axis=2) > np.max(ZLCFSframe)
    mask = np.logical_or(np.logical_or(np.logical_or(cond1,cond2), psinormRZ_dense>1), np.isnan(psinormRZ_dense))
    psinormRZ_masked = np.ma.masked_array(psinormRZ_dense, mask=mask, fill_value=0.0)


    # plot masked flux surfaces 
    psi.contourf(Rdense,Zdense,psinormRZ_masked[t_idx,:,:])


    plt.figure()
    plt.imshow(psinormRZ_masked[t_idx,:,:])
    plt.colorbar()

    return psinormRZ_masked

if __name__ == '__main__':
    psi = get_psi_map()
