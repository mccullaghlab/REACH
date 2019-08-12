
import numpy as np
import MDAnalysis as md
# load reach_routines.py as library
import reach_routines as reach

# initialize MDAnalysis universe for selection purposes
topFile = "IGPS_apo_v02_1us_com_average_structure.pdb"
coord = md.Universe(topFile)
sel = coord.select_atoms("all")

# setup REACH selections if you have multiple molecules or want to specify secondary structure elements etc
mols = ["resid 1:253","resid 254:454"] 
nSels = len(mols)
reachSels= []
for i in range(nSels):
    reachSels.append(coord.select_atoms(mols[i]))

# load covariance matrix and average position
covar = np.loadtxt("IGPS_apo_v02_1us_com_covar.dat")
avgPos = np.loadtxt("IGPS_apo_v02_1us_com_average_structure.dat")

# compute reach Hessian
reachHessian = reach.perform_reach(covar,avgPos,reachSels)
np.savetxt("reach_hessian.dat",reachHessian)





