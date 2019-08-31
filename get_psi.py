
import numpy as np
import eqtools
import matplotlib.pyplot as plt
import sys
import MDSplus

import scipy
import matplotlib.gridspec as mplgs

plt.ion()


root = '\\analysis::top.efit.results.'
gfile = 'g_eqdsk'


tree = MDSplus.Tree('analysis', 1101014019)
psiNode = tree.getNode(root+gfile+':psirz')
psiRZ = psiNode.data()
rGrid = psiNode.dim_of(0).data()
zGrid = psiNode.dim_of(1).data()



fluxPlot = plt.figure(figsize=(6,11))
gs = mplgs.GridSpec(2,1,height_ratios=[30,1])
psi = fluxPlot.add_subplot(gs[0,0])
xlim = psi.get_xlim()
ylim = psi.get_ylim()
            
# dummy plot to get x,ylims
psi.contourf(rGrid,zGrid,psiRZ[0])# ,1)
