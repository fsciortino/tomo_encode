
import numpy as np
import eqtools
import matplotlib.pyplot as plt
import sys
import MDSplus
import eqtools
from IPython import embed
import scipy
import matplotlib.gridspec as mplgs

plt.ion()



def get_psi_map(e,plot=True,t_val=None):
    ''' Get Psi map only in confined region. 
    The first argument is an 'eqtools' tree, the second the time [s] to get psi map at. 
    '''
    if plot and t_val is None:
        print('Plotting at t=1.2 s')
        t_val = 1.2
        
    times = e.getTimeBase()

    # get major radius on axis 
    R0 = e.getMagR()

    # flux on R,Z grids
    psiRZ = e.getFluxGrid()
    rGrid = e.getRGrid(length_unit='m')
    zGrid = e.getZGrid(length_unit='m')

    # coordinates along LCFS
    RLCFS = e.getRLCFS(length_unit='m')
    ZLCFS = e.getZLCFS(length_unit='m')

    # find center (largest flux)
    R0 = e.getMagR()
    Z0 = e.getMagZ()
    psiRZ_masked = np.zeros_like(psiRZ)

    # interpolate psiRZ on denser grid
    Rdense = np.linspace(np.min(rGrid), np.max(rGrid),200)
    Zdense = np.linspace(np.min(zGrid), np.max(zGrid),250)

    RR,ZZ = np.meshgrid(Rdense,Zdense)
    psinormRZ_dense = e.rz2phinorm(RR,ZZ,times,sqrt=True)  #normalized sqrt of poloidal flux coordinate

    # conditions used to identify the confined plasma region at every time step
    cond1 = np.stack([ZZ<np.min(ZLCFS[t_idxx,\
                                      scipy.where(scipy.logical_or(RLCFS[t_idxx,:]>0.0,scipy.isnan(RLCFS[t_idxx,:])))[0]])\
                      for t_idxx in range(len(times))])
    cond2 = np.stack([ZZ>np.max(ZLCFS[t_idxx,\
                                      scipy.where(scipy.logical_or(RLCFS[t_idxx,:] > 0.0,scipy.isnan(RLCFS[t_idxx,:])))[0]])\
                      for t_idxx in range(len(times))])
    
    mask = np.logical_or(np.logical_or(np.logical_or(cond1,cond2), psinormRZ_dense>1), np.isnan(psinormRZ_dense))
    psinormRZ_masked = np.ma.masked_array(psinormRZ_dense, mask=mask, fill_value=0.0)

    if plot:
        # plot at a specific time, given by the user
        t_idx = np.argmin(np.abs(times - t_val))

        maskarr = scipy.where(scipy.logical_or(RLCFS[t_idxx,:]>0.0,scipy.isnan(RLCFS[t_idxx,:])))[0]
        RLCFSframe = RLCFS[t_idx,maskarr]
        ZLCFSframe = ZLCFS[t_idx,maskarr]
    
        fluxPlot = plt.figure(figsize=(6,11))
        gs = mplgs.GridSpec(2,1,height_ratios=[30,1])
        psi = fluxPlot.add_subplot(gs[0,0])
        xlim = psi.get_xlim()
        ylim = psi.get_ylim()

        # dummy plot to get x,ylims
        psi.contour(rGrid,zGrid,psiRZ[0],1)

        # add LCFS contour
        psi.plot(RLCFSframe,ZLCFSframe,'r',linewidth=2.0,zorder=3)
        
        # plot masked flux surfaces 
        psi.contourf(Rdense,Zdense,psinormRZ_masked[t_idx,:,:])
        
        # plot separately the masked flux surfaces
        plt.figure()
        plt.imshow(psinormRZ_masked[t_idx,:,:])
        plt.colorbar()

    return psinormRZ_masked


class IndexTracker(object):
    def __init__(self, ax, fig, X, times):
        self.ax = ax
        self.times = times
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:,:,self.ind]) 
        self.cb = fig.colorbar(self.im)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.cb.remove()
        self.im.set_data(self.X[:,:,self.ind])
        ax.set_title('slice %s, time = %f' %(self.ind,self.times[self.ind]))
        self.cb = fig.colorbar(self.im)
        self.im.axes.figure.canvas.draw()

        

if __name__ == '__main__':

    shot=1101014019
    e = eqtools.CModEFITTree(shot)
    
    psi = get_psi_map(e, t_val=1.2)

    times = e.getTimeBase()

    # create visualization of the time evolution of the confined region (scroll with mouse to move in time)
    fig,ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax,fig, np.transpose(psi,axes=(1,2,0)), times)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)


    

    
