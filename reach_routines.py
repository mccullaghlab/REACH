# load libraries
import numpy as np
from scipy.optimize import curve_fit
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.distances import distance_array
import matplotlib.pyplot as plt
from numba import jit

kb = 1.9872036E-3 # kcal/mol/K

def sigmoid(r,a,b,c):
    return a * ( 1 - 1/(1+np.exp(-b*(r-c))) )
def exponential(r,a,b):
    return a*np.exp(-b*r)

def iterative_align_average(coord,selGroup,frameStart=0,frameStop=-1,deltaFrame=1,maxSteps=25,thresh=0.001):
    if frameStop < 0:
        frameStop = coord.trajectory.n_frames + frameStop + 1
    nFrames= int((frameStop-frameStart)//deltaFrame)
    # create numpy array of aligned positions
    alignedPos = np.empty((nFrames,selGroup.n_atoms,3),dtype=np.float64)
    #generate an initial average by aligning to first frame
    avg = np.zeros((selGroup.n_atoms,3),dtype=np.float64)
    frameCount = 0
    for ts in coord.trajectory[frameStart:frameStop:deltaFrame]:
        selGroup.translate(-selGroup.center_of_mass())
        if frameCount == 0:
            ref = np.copy(selGroup.positions)
        else:
            R = align.rotation_matrix(selGroup.positions, ref)[0]
            selGroup.rotate(R)
        avg += selGroup.positions
        alignedPos[frameCount,:,:] = selGroup.positions
        frameCount += 1
    # finish average
    avg /= nFrames
    # perform iterative alignment and average to converge average
    newAvg = np.zeros((selGroup.n_atoms,3),dtype=np.float64)
    avgRmsd = 2*thresh
    step = 0
    while step<maxSteps and avgRmsd > thresh:
        newAvg = 0.0
        frameCount = 0
        for ts in coord.trajectory[frameStart:frameStop:deltaFrame]:
            alignedPos[frameCount,:,:] -= np.mean(alignedPos[frameCount,:,:],axis=0)
            R = align.rotation_matrix(alignedPos[frameCount,:,:], avg)[0]
            alignedPos[frameCount,:,:] = np.dot(alignedPos[frameCount,:,:],R.T)
            newAvg += alignedPos[frameCount,:,:]
            frameCount += 1
        # finish average
        newAvg /= nFrames
        avgRmsd = rmsd(avg,newAvg,center=False,superposition=False)
        avg = np.copy(newAvg)
        step += 1
        print(step, avgRmsd)
    return avg, alignedPos

# routine to compute covariance and Hessian for a selection from a trajectory
# coord is an MDAnalysis universe
# sel is an MDAnalysis atom group
def perform_reach_traj(coord,sel,nWindows=1,deltaFrame=1,thresh=1E-2):
    # setup some variables for windows and trajectory
    totalFrames = coord.trajectory.n_frames
    windowFrames = totalFrames//nWindows
    # declare arrays
    covar = np.zeros((3*sel.n_atoms,3*sel.n_atoms),dtype=np.float64)
    hessian = np.zeros((sel.n_atoms,sel.n_atoms),dtype=np.float64)
    pairDist = np.zeros((sel.n_atoms,sel.n_atoms),dtype=np.float64)
    totalFrameCount = 0
    # loop through trajectory in windows
    for window in range(nWindows):
        print("Window:", window)
        startFrame = window*windowFrames
        stopFrame = startFrame+windowFrames
        # start with total selection
        # perform iterative alignment to average and compute converged average structure
        avgPos, alignedPos = iterative_align_average(coord,sel,frameStart=startFrame,frameStop=stopFrame,deltaFrame=deltaFrame,thresh=thresh)
        # zero some loop variables
        frameCount = 0
        covar = 0.0
        # loop through trajectory and compute covariance 
        for ts in coord.trajectory[startFrame:stopFrame:deltaFrame]:
            covar += np.dot(alignedPos[frameCount,:,:].reshape(3*sel.n_atoms,1),alignedPos[frameCount,:,:].reshape(1,3*sel.n_atoms))
            pairDist += distance_array(alignedPos[frameCount,:,:],alignedPos[frameCount,:,:],box=ts.dimensions)
            frameCount += 1
            totalFrameCount += 1
        # finish covariance
        covar /= frameCount
        covar -= np.dot(avgPos.reshape(3*sel.n_atoms,1),avgPos.reshape(1,3*sel.n_atoms))
        # compute hessian
        hessian3D = hessian_from_covar(covar,303)
        hessian += hessian_NxN_from_3Nx3N(hessian3D)
    # finish pairwise distance average
    pairDist /= totalFrameCount
    hessian /= nWindows
    return pairDist, hessian, covar

def hessian_NxN_from_3Nx3N(hessian):
    N3 = hessian.shape[0]
    N = N3//3
    k = np.empty((N,N),dtype=np.float64)
    for i in range(N):
        iIndex = [i*3,i*3+1,i*3+2]
        for j in range(N):
            jIndex = [j*3,j*3+1,j*3+2]
            k[i,j] = np.trace(hessian[i*3:i*3+3,j*3:j*3+3])
    return k    

def hessian_from_covar(covar,T):
    # diagonalize covariance
    delta, V = np.linalg.eigh(covar)
    # sort eigenvalues
    idx = abs(delta).argsort()
    delta = delta[idx]
    V = V[:,idx]
    # compute reciprocal eigenvalue diagonal matrix
    deltaInv = np.diag(1.0/delta)
    # remove COM translation and rotation (assuming 3D)
    for i in range(6):
        deltaInv[i,i] = 0.0
    # compute hessian
    hessian = kb*T*np.dot(V,np.dot(deltaInv,V.T))
    return hessian


def compile_nb_dists_ks(pairDist,hessian,minLim,selGroup):
    nAtoms = pairDist.shape[0]
    dist = []
    k = []
    nSels = len(selGroup)
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-minLim-1):
            for j in range(i+minLim-1,selGroup[n].n_atoms):
                dist.append(pairDist[i,j])
                k.append(-hessian[i,j])
        start += selGroup[n].n_atoms
    nStart = 0
    for n in range(nSels-1):
        mStart = nStart + selGroup[n].n_atoms
        for m in range(n+1,nSels):
            for i in range(nStart,nStart + selGroup[n].n_atoms):
                for j in range(mStart,mStart + selGroup[m].n_atoms):
                    dist.append(pairDist[i,j])
                    k.append(-hessian[i,j])
            mStart += selGroup[m].n_atoms
        nStart += selGroup[n].n_atoms
                
    return dist, k

def compile_dists_ks(pairDist,hessian,minLim,selGroup):
    nAtoms = pairDist.shape[0]
    dist = []
    k = []
    nSels  = len(selGroup)
    start = 0
    for j in range(nSels):
        for i in range(start,start + selGroup[j].n_atoms - minLim - 1):
            dist.append(pairDist[i,i+minLim-1])
            k.append(-hessian[i,i+minLim-1])
        start += selGroup[j].n_atoms
    return dist, k

def bin_ks(dist,k,minX=0.0,maxX=20.0,binSize=0.4,fitFunction=sigmoid):
    x = np.arange(minX,maxX,binSize)
    avgK = np.zeros(x.size,dtype=np.float64)
    varK = np.zeros(x.size,dtype=np.float64)
    count = np.zeros(x.size,dtype=int)
    for i in range(len(k)):
        binK = int(dist[i]/binSize)
        if binK > 0 and binK < x.size:
            avgK[binK] += k[i]
            varK[binK] += k[i]**2
            count[binK] += 1
    for i in range(x.size):
        if count[i] > 0:
            avgK[i] /= count[i]
            varK[i] /= count[i]
            varK[i] = np.sqrt((varK[i]-avgK[i]**2)/count[i])

    #lstq_sig = curve_fit(fitFunction,x[np.nonzero(count)],avgK[np.nonzero(count)])
    
    data = np.column_stack((x,avgK,varK,count))
    
    return data#, lstq_sig

def fit_data(dist,k,fitFunction=sigmoid):

    lstq_sig = curve_fit(fitFunction,dist,k)
    
    return lstq_sig


# sel is an MDAnalysis atom group
def perform_reach_traj_com(coord,sel,nWindows=1,deltaFrame=1,thresh=1E-2):
    # setup some variables for windows and trajectory
    totalFrames = coord.trajectory.n_frames
    windowFrames = totalFrames//nWindows
    # declare arrays
    covar = np.zeros((3*sel.n_residues,3*sel.n_residues),dtype=np.float64)
    hessian = np.zeros((sel.n_residues,sel.n_residues),dtype=np.float64)
    pairDist = np.zeros((sel.n_residues,sel.n_residues),dtype=np.float64)
    totalFrameCount = 0
    # loop through trajectory in windows
    for window in range(nWindows):
        print("Window:", window)
        startFrame = window*windowFrames
        stopFrame = startFrame+windowFrames
        # start with total selection
        # perform iterative alignment to average and compute converged average structure
        avgPos, alignedPos = iterative_align_average_com(coord,sel,frameStart=startFrame,frameStop=stopFrame,deltaFrame=deltaFrame,thresh=thresh)
        # zero some loop variables
        frameCount = 0
        covar = 0.0
        print("Creating Covariance")
        # loop through trajectory and compute covariance 
        for ts in coord.trajectory[startFrame:stopFrame:deltaFrame]:
            covar += np.dot(alignedPos[frameCount,:,:].reshape(3*sel.n_residues,1),alignedPos[frameCount,:,:].reshape(1,3*sel.n_residues))
            pairDist += distance_array(np.float32(alignedPos[frameCount,:,:]),np.float32(alignedPos[frameCount,:,:]),box=ts.dimensions)
            frameCount += 1
            totalFrameCount += 1
        # finish covariance
        covar /= frameCount
        covar -= np.dot(avgPos.reshape(3*sel.n_residues,1),avgPos.reshape(1,3*sel.n_residues))
        # compute hessian
        hessian3D = hessian_from_covar(covar,303)
        hessian += hessian_NxN_from_3Nx3N(hessian3D)
    # finish pairwise distance average
    pairDist /= totalFrameCount
    hessian /= nWindows
    return pairDist, hessian, covar

@jit
def iterative_align_average_com(coord,selGroup,frameStart=0,frameStop=-1,deltaFrame=1,maxSteps=25,thresh=0.001):
    if frameStop < 0:
        frameStop = coord.trajectory.n_frames + frameStop + 1
    nFrames= int((frameStop-frameStart)//deltaFrame)
    # create numpy array of aligned positions
    alignedPos = np.empty((nFrames,selGroup.n_residues,3),dtype=np.float64)
    #generate an initial average by aligning to first frame
    avg = np.zeros((selGroup.n_residues,3),dtype=np.float64)
    comPos = np.empty((selGroup.n_residues,3),dtype=np.float64)
    frameCount = 0
    for ts in coord.trajectory[frameStart:frameStop:deltaFrame]:
        selGroup.translate(-selGroup.center_of_mass())
        for i, resid in enumerate(np.unique(selGroup.resids)):
            residSel = "resid " + str(resid)
            comPos[i,:] = selGroup.select_atoms(residSel).center_of_mass()            
        if frameCount == 0:
            ref = np.copy(comPos)
        else:
            R = align.rotation_matrix(comPos, ref)[0]
            comPos = np.dot(comPos,R.T)
        avg += comPos
        alignedPos[frameCount,:,:] = comPos
        frameCount += 1
    # finish average
    avg /= nFrames
    # perform iterative alignment and average to converge average
    newAvg = np.zeros((selGroup.n_residues,3),dtype=np.float64)
    avgRmsd = 2*thresh
    step = 0
    while step<maxSteps and avgRmsd > thresh:
        newAvg = 0.0
        frameCount = 0
        for ts in coord.trajectory[frameStart:frameStop:deltaFrame]:
            alignedPos[frameCount,:,:] -= np.mean(alignedPos[frameCount,:,:],axis=0)
            R = align.rotation_matrix(alignedPos[frameCount,:,:], avg)[0]
            alignedPos[frameCount,:,:] = np.dot(alignedPos[frameCount,:,:],R.T)
            newAvg += alignedPos[frameCount,:,:]
            frameCount += 1
        # finish average
        newAvg /= nFrames
        avgRmsd = rmsd(avg,newAvg,center=False,superposition=False)
        avg = np.copy(newAvg)
        step += 1
        print(step, avgRmsd)
    return avg, alignedPos

def reachify_hessian(hessian,pairDist,selGroup):
    
    dist14, k14 = compile_dists_ks(pairDist,hessian,4,selGroup)
    #data14, lstq14 = bin_ks(dist14,k14,binSize=0.01,fitFunction=exponential)
    exp14 = fit_data(dist14,k14,fitFunction=exponential)
    
    dist15, k15 = compile_dists_ks(pairDist,hessian,5,selGroup)
    #data15, lstq15 = bin_ks(dist15,k15,binSize=0.1)
    sig15 = fit_data(dist15,k15,fitFunction=sigmoid)
    
    distNB, kNB = compile_nb_dists_ks(pairDist,hessian,6,selGroup)
    #dataNB, lstqNB = bin_ks(distNB,kNB,binSize=0.2,fitFunction=sigmoid)
    expNB = fit_data(distNB,kNB,fitFunction=exponential)
    #expNB = [ [1980.0,0.749] ]
    #expNB[0][0] = 1980.0
    #expNB[0][1] = 0.749
    
    # generate new k matrix
    N = hessian.shape[0]
    k_matrix = np.zeros((N,N),dtype=np.float64)
    nSels = len(selGroup)
    # 1-2 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-1):
            k_matrix[i,i+1] = k_matrix[i+1,i] = -hessian[i,i+1]
        start += selGroup[n].n_atoms
    # 1-4 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-3):
            k_matrix[i,i+3] = k_matrix[i+3,i] = exponential(pairDist[i,i+3],exp14[0][0],exp14[0][1])
        start += selGroup[n].n_atoms
    # 1-5 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-4):
            k_matrix[i,i+4] = k_matrix[i+4,i] = sigmoid(pairDist[i,i+4],sig15[0][0],sig15[0][1],sig15[0][2])
        start += selGroup[n].n_atoms
    # non-bonded (beyond 1-5)
    # non-bonded within each selection
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms):
            for j in range(i+5,start + selGroup[n].n_atoms):
                k_matrix[i,j] = k_matrix[j,i] = exponential(pairDist[i,j],expNB[0][0],expNB[0][1])
        start += selGroup[n].n_atoms
    # non-bonded between each selection (assumed to be all terms in off coupling sub-matrix)
    nStart = 0
    for n in range(nSels-1):
        mStart = nStart + selGroup[n].n_atoms
        for m in range(n+1,nSels):
            for i in range(nStart,nStart + selGroup[n].n_atoms):
                for j in range(mStart,mStart + selGroup[m].n_atoms):
                    k_matrix[i,j] = k_matrix[j,i] = exponential(pairDist[i,j],expNB[0][0],expNB[0][1])
            mStart += selGroup[m].n_atoms
        nStart += selGroup[n].n_atoms
    # add diagonals
    new_hessian = -np.copy(k_matrix)
    for i in range(N):
        new_hessian[i,i] = np.sum(k_matrix[:,i])
    return new_hessian, [exp14,sig15,expNB]

def plot_nb(hessian,pairDist,selGroup):
       
    distNB, kNB = compile_nb_dists_ks(pairDist,hessian,6,selGroup)
    #dataNB, lstqNB = bin_ks(distNB,kNB,binSize=0.2,fitFunction=sigmoid)
    #expNB = fit_data(distNB,kNB,fitFunction=exponential)
        # setup plot parameters
    fig = plt.figure(figsize=(10,8), dpi= 80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
    ax.set_xlabel("Distance",size=20)
    ax.set_ylabel("k",size=20)
    plt.tick_params(axis='both',labelsize=20)
    ax.plot(distNB,kNB,'o')

def reachify_hessian_prefit(hessian,pairDist,selGroup,fits):
    
    # generate new k matrix
    N = hessian.shape[0]
    k_matrix = np.zeros((N,N),dtype=np.float64)
    nSels = len(selGroup)
    # 1-2 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-1):
            k_matrix[i,i+1] = k_matrix[i+1,i] = -hessian[i,i+1]
        start += selGroup[n].n_atoms
    # 1-4 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-3):
            k_matrix[i,i+3] = k_matrix[i+3,i] = exponential(pairDist[i,i+3],fits[0][0][0],fits[0][0][1])
        start += selGroup[n].n_atoms
    # 1-5 interactions
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms-4):
            k_matrix[i,i+4] = k_matrix[i+4,i] = sigmoid(pairDist[i,i+4],fits[1][0][0],fits[1][0][1],fits[1][0][2])
        start += selGroup[n].n_atoms
    # non-bonded (beyond 1-5)
    # non-bonded within each selection
    start = 0
    for n in range(nSels):
        for i in range(start,start + selGroup[n].n_atoms):
            for j in range(i+5,start + selGroup[n].n_atoms):
                k_matrix[i,j] = k_matrix[j,i] = exponential(pairDist[i,j],fits[2][0][0],fits[2][0][1])
        start += selGroup[n].n_atoms
    # non-bonded between each selection (assumed to be all terms in off coupling sub-matrix)
    nStart = 0
    for n in range(nSels-1):
        mStart = nStart + selGroup[n].n_atoms
        for m in range(n+1,nSels):
            for i in range(nStart,nStart + selGroup[n].n_atoms):
                for j in range(mStart,mStart + selGroup[m].n_atoms):
                    k_matrix[i,j] = k_matrix[j,i] = exponential(pairDist[i,j],fits[2][0][0],fits[2][0][1])
            mStart += selGroup[m].n_atoms
        nStart += selGroup[n].n_atoms
    # add diagonals
    new_hessian = -np.copy(k_matrix)
    for i in range(N):
        new_hessian[i,i] = np.sum(k_matrix[:,i])
    return new_hessian    
    
