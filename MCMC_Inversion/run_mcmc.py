# -*- coding: utf-8 -*-
"""
A Python 2.7 implementation of the DREAMzs MCMC sampler (Vrugt et al., 2009, 
Laloy and Vrugt, 2012) with CRN application considered in Laloy et al. (2017). 
This DREAMzs implementation is based on the 2013 DREAMzs Matlab code (version 1.5, 
licensed under GPL3) written by Jasper Vrugt (FYI: a more recent Matlab p-code 
with many more options is available at http://faculty.sites.uci.edu/jasper/). 

Version 0.0 - January 2017. Probably a bit non-pythonic coding and not optimized for 
speed, but the does the job and the forward model evaluations can be performed in 
parallel on several CPUs (not recommended for this CRN application as the forward 
model is very quick). Also, this "run_mcmc" script is separated in "#%%" sections 
that can conveniently be run separately with Spyder.

@author: Eric Laloy <elaloy@sckcen.be>

Please drop me an email if you have any question and/or if you find a bug in this
program. 

Also, if you find this code useful please cite the paper for which it has been 
implemented (Laloy et al., 2017) as well as the DREAMzs paper(s).

===
Copyright (C) 2017  Eric Laloy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ===                               

References:
    
Laloy, E., Beertem, K., Vanacker, V., Christl, M., Rogiers, B., Wauters, L., 
    Bayesian inversion of a CRN depth profile to infer Quaternary erosion of the 
    northwestern Campine Plateau (NE Belgium), Earth Surf. Dynam., 5, 331–345, 
    2017, https://doi.org/10.5194/esurf-5-331-2017.
    
Laloy, E., Vrugt, J.A., High-dimensional posterior exploration of hydrologic models      
    using multiple-try DREAMzs and high-performance computing, Water Resources Research, 
    48, W01526, doi:10.1029/2011WR010608, 2012.
    
ter Braak, C.J.F., Vrugt, J.A., Differential Evolution Markov Chain with snooker updater 
    and fewer chains, Statistics and Computing, 18, 435–446, doi:10.1007/s11222-008-9104-9,
	2008.
    
Vrugt, J. A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and J.M. Hyman,
    Accelerating Markov chain Monte Carlo simulation by differential evolution with
    self-adaptive randomized subspace sampling, International Journal of Nonlinear Sciences
    and Numerical Simulation, 10(3), 273-290, 2009.                                         
                                                                                                                                                                                                       
"""

import os
import time

main_dir=r'D:\CRN_ProbabilisticInversion\MCMC_Inversion' # Set the working directory

os.chdir(main_dir)

import mcmc
import numpy as np

#% Set rng_seed and case study

rng_seed=1 # np.random.seed(np.floor(time.time()).astype('int'))

CaseStudy=2
 
if  CaseStudy==0: #100-d correlated gaussian (case study 2 in DREAMzs Matlab code)
    seq=3
    steps=5000
    ndraw=seq*100000
    thin=10
    jr_scale=1.0
    Prior='LHS'

if  CaseStudy==1: #10-d bimodal distribution (case study 3 in DREAMzs Matlab code)
    seq=5
    ndraw=seq*40000
    thin=10
    steps=np.int32(ndraw/(20.0*seq))
    jr_scale=1.0
    Prior='COV'
    
if  CaseStudy==2: # CRN data inversion (see Laloy et al. (ESURF 2017))
    seq=5
    ndraw=seq*30000
    thin=1
    steps=500
    jr_scale=1.25
    Prior='Prior_CRN_1' 

#% Run the DREAMzs algorithm
if __name__ == '__main__':
    
    start_time = time.time()
    
    q=mcmc.Sampler(CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,Prior=Prior,parallel_jobs=seq,steps=steps,
                   parallelUpdate = 0.9,pCR=True,thin=thin,nCR=3,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Reflect',
                   lik_sigma_est=False,DoParallel=False,jr_scale=jr_scale,rng_seed=rng_seed)
    
    print("Iterating")
    
    tmpFilePath=None # None or: main_dir+'\out_tmp.pkl'
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar = q.sample(RestartFilePath=tmpFilePath)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
    
#%% Visualization of results
#import os
#main_dir=r'D:\CRN_ProbabilisticInversion\MCMC_Inversion'
#os.chdir(main_dir)
#import numpy as np 
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)

with open('dreamzs_out'+'.pkl','rb') as f: # Use this to load final results only
#with open('out_tmp'+'.pkl','rb') as f: # Use this to load temporary or final results                                  
                                        # together with every used variable
    tmp_obj=pickle.load(f)
try:
    Sequences=tmp_obj['Sequences']
    Z=tmp_obj['Z']
    OutDiag=tmp_obj['OutDiag']
    fx=tmp_obj['fx']
    MCMCPar=tmp_obj['MCMCPar']
    Measurement=tmp_obj['Measurement']
    Extra=tmp_obj['Extra']
    MCMCVar=tmp_obj['MCMCVar']
    Modelname=tmp_obj['ModelName']
    print(MCMCVar.Iter)
except:
    pass
del tmp_obj



from mcmc_func import Genparset
# Uncomment the next 2 lines if temporary results are loaded (need to remove zeros)
# idx=np.argwhere(Sequences[:,0,0]!=0) 
# Sequences=Sequences[idx[:,0],:,:]

ParSet=Genparset(Sequences)

#%% Check   acceptance rate: should be in 10% - 50% or so
AR = np.mean(OutDiag.AR[1:,1]);print(AR)
fig=plt.figure
plt.plot(OutDiag.AR[1:,0],OutDiag.AR[1:,1],'ob')
#%% Check Rstat convergence
dummy=np.where((OutDiag.R_stat[:,1:]<=1.2) & (OutDiag.R_stat[:,1:] > 0))
try:
    row_num, counts = np.unique(dummy[0],return_counts=True)
    row_num=row_num[np.argmax(counts==MCMCPar.n)]
    print('R_stat convergence declared at iteration '+str(OutDiag.R_stat[row_num,0]))
except:
    print('Not converged according to R_stat')
    pass
#%% Compare simulated values against measurements
ii=np.where(ParSet[:,-2]==np.min(ParSet[:,-2]))
ii=ii[0][0]
print(ParSet[ii,-2])
ii=np.where(ParSet[:,-1]==np.max(ParSet[:,-1]))
ii=ii[0][0]
print(ParSet[ii,-2])
xx=np.linspace(8e4,1.6e5,1000)
fig=plt.figure
plt.plot(xx,xx,'-k');plt.hold(True)
plt.plot(Measurement.MeasData[0,:],fx[ii,:],'ob')
#%% Plot marginal posterior distributions
from scipy.stats import norm

filename='MarginalDistributionsMCMC'
savefigure=False
xlabel=[r'$E \rm \ [m/Myr]$',r'$t \rm \ [yr]$',r'${N}_{\rm inh}\rm \ [atoms/g]$',
        r'$\sigma_{\rm e}\rm \ [atoms/g]$']
ylabel=r'$\rm Density$'
sub_letter=['(a)','(b)','(c)','(d)']
#xti=[np.array([0,20,40,60]),np.array([0,5e5,1e6]),np.array([10000,50000,90000]),
#     np.array([5000,10000,15000,20000,25000])]
     
xti=[np.array([0,20,40,60]),np.array([0,5e5,1e6]),np.array([10000,50000,90000]),
     np.array([5000,10000,15000,20000,25000])]
nbins=30

fig=plt.figure(figsize=(11,11))

for i in xrange(0,MCMCPar.n):
    if i < MCMCPar.n-1:
        x=ParSet[3000:,i]
        bins = np.linspace(MCMCPar.lb[0,i], MCMCPar.ub[0,i], nbins)
    else:
        x=10**ParSet[3000:,i]
        bins = np.linspace(10**MCMCPar.lb[0,i], 10**MCMCPar.ub[0,i], nbins)
  
    yv,xv=np.histogram(x,bins=bins,density=True)
    delta=np.diff(xv)[0]
    # sum(yv*delta) should be equal to 1
    xv=xv[1:]-0.5*(xv[1]-xv[0])
    sub=plt.subplot(2,2,i+1)
    # Plot posterior
    plt.bar(xv,yv,width=0.95*delta, color='b')
    plt.hold(True)
    # And now plot prior
    if i < MCMCPar.n-1:
        xx=np.linspace(MCMCPar.lb[0,i]*0.9,MCMCPar.ub[0,i]*1.1,1e4)
    else:
        xx=np.linspace(0.9*(10**MCMCPar.lb[0,i]), 1.1*(10**MCMCPar.ub[0,i]), 1e4)
    if i==0: # Normal prior
        yy=norm.pdf(xx,MCMCPar.pmu[i],MCMCPar.psd[i])
    elif i==1: # Uniform prior
        yy=np.zeros(len(xx))++1/(MCMCPar.ub[0,i]-MCMCPar.lb[0,i])
    elif i ==2: # Uniform prior
        yy=np.zeros(len(xx))++1/(MCMCPar.ub[0,i]-MCMCPar.lb[0,i])
    elif i==3: # Log-uniform (Jeffreys) prior
        yy=1/xx;dxx=np.diff(xx);dxx=np.concatenate((np.array([dxx[0]]),dxx))
        yy=yy/(np.sum(dxx*yy))
    plt.plot(xx,yy,'-r');plt.hold(False)   
       
    # Labels, limits and others
    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    sub.axes.set_xlabel(xlabel[i], fontsize = 18.0)
    sub.axes.get_xaxis().set_ticks(xti[i])
    sub.axes.tick_params(labelsize=12)
    if i < MCMCPar.n-1:
        sub.axes.set_xlim(MCMCPar.lb[0,i], MCMCPar.ub[0,i])
    else:
        sub.axes.set_xlim(10**MCMCPar.lb[0,i], 10**MCMCPar.ub[0,i])
    if (i==0 or i==2):
        sub.axes.set_ylabel(ylabel, fontsize = 18.0)
        sub.text(0.04, 0.92, sub_letter[i], transform=sub.axes.transAxes, 
            size=16)
    else:
        sub.text(0.90, 0.92, sub_letter[i], transform=sub.axes.transAxes, 
            size=18)
    
plt.tight_layout()    
plt.show()    
if savefigure==True:
    fig.savefig(filename+'.png',dpi=600)
#%% PLot of total erosion distribution (E x t)
filename='TotalErosionMCMC'
savefigure=False
istart=1500
etot=ParSet[istart:,0]*1e-6*ParSet[istart:,1]
xlabel=r'$\rm Total\ erosion\ [m]$'
ylabel=r'$\rm Density$'
nbins=20
fig=plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.hist(etot,nbins,normed=1,rwidth=1)
ax.set_xlabel(xlabel, fontsize = 20.0)
ax.set_ylabel(ylabel, fontsize = 20.0)
ax.tick_params(labelsize=14)

if savefigure==True:
    fig.savefig(filename+'.png',dpi=600)

#%% Find mode of Marginal posteriors (rounded to nearest multiple of xx):
def roundup(x,a):
    return np.round(x / float(a)).astype('int') * int(a)   
from scipy.stats import mode
xx=np.array([1,50000,5000,500])
for i in xrange(0,MCMCPar.n):
    if i<3:
        dum=mode(roundup(ParSet[5000:,i],xx[i]))
    else:
        dum=mode(roundup(10**ParSet[5000:,i],xx[i]))
    print(dum)    
#%% 2D scatter plots + iso-density contours
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import stats
sub_letter=['(a)','(b)','(c)']
positxt=np.array([[0.04,0.91],[0.04,0.05],[0.04,0.05]])
filename='ScatAndContPLotsMCMC'
savefigure=False
DoThinning=True
mycmap=cm.autumn

fig = plt.figure(figsize=(12,10))
gs = gridspec.GridSpec(2, 4)
for i in xrange(0,MCMCPar.n-1):
  
    if i==0:
        sub=plt.subplot(gs[0, 0:2])
        xx=ParSet[3000:,0]
        yy=ParSet[3000:,1]
        xlabel=r'$E \rm \ [m/Myr]$'
        ylabel=r'$t \rm \ [yr]$'
        if DoThinning==True:
            idx=np.arange(0,xx.shape[0],10)
            xx=xx[idx]
            yy=yy[idx]
        # Set ranges for kernel density smoothing
        xmin=MCMCPar.lb[0,0];xmax=MCMCPar.ub[0,0]
        ymin=MCMCPar.lb[0,1];ymax=MCMCPar.ub[0,1]
       
    if i==1:
        sub=plt.subplot(gs[0,2:])
        xx=ParSet[3000:,0]
        yy=ParSet[3000:,2]
        xlabel=r'$E \rm \ [m/Myr]$'
        ylabel=r'${N}_{\rm inh}\rm \ [atoms/g]$'
        if DoThinning==True:
            idx=np.arange(0,xx.shape[0],10)
            xx=xx[idx]
            yy=yy[idx]
        # Set ranges for kernel density smoothing
        xmin=MCMCPar.lb[0,0];xmax=MCMCPar.ub[0,0]
        ymin=MCMCPar.lb[0,2];ymax=MCMCPar.ub[0,2]
       
    if i==2:
        sub=plt.subplot(gs[1, 1:3])
        xx=ParSet[3000:,1]
        yy=ParSet[3000:,2]
        xlabel=r'$t \rm \ [yr]$'
        ylabel=r'${N}_{\rm inh}\rm \ [atoms/g]$'
        if DoThinning==True:
            idx=np.arange(0,xx.shape[0],10)
            xx=xx[idx]
            yy=yy[idx]
        # Set ranges for kernel density smoothing
        xmin=MCMCPar.lb[0,1];xmax=MCMCPar.ub[0,1]
        ymin=MCMCPar.lb[0,2];ymax=MCMCPar.ub[0,2]
       
    # Perform kernel density smoothing
    xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xi.ravel(), yi.ravel()])
    values = np.vstack([xx, yy])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xi.shape)
       
    plt.scatter(xx,yy,c='k',marker=".",zorder=-1)
    plt.hold(True)
    plt.contour(xi, yi, f,6,linewidths=np.zeros((6))+1.5,cmap=mycmap)
    plt.hold(False)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    sub.axes.set_xlabel(xlabel, fontsize=20)  
    sub.axes.set_ylabel(ylabel, fontsize=20)
    sub.text(positxt[i,0], positxt[i,1], sub_letter[i], transform=sub.axes.transAxes, 
            size=17, weight='normal')
    sub.axes.tick_params(labelsize=12)
    plt.tight_layout()
    
if savefigure==True:
    fig.savefig(filename+'.png',dpi=600)

#%% Look at correlations between parameters
cc=np.corrcoef(ParSet[5000:,:MCMCPar.n].transpose())
print(cc)

#%% Recompute the Gelman and Rubin (1992) criterion with a given fraction of 
#   past samples if wished

from mcmc_func import GelmanRubin
past_frac=0.3
Rstat=np.zeros((1,MCMCPar.n))
m,n,p=Sequences.shape
for i in range(0,m,MCMCPar.steps*2):
    start_idx = np.maximum(1,np.floor(past_frac*i)).astype('int64')-1; end_idx = i
  
    current_R_stat = GelmanRubin(Sequences[start_idx:end_idx,:MCMCPar.n,:MCMCPar.seq],MCMCPar)
    Rstat=np.append(Rstat,current_R_stat.reshape((1,MCMCPar.n)),axis=0)
#%% Predictive uncertainty plots
# Sample npar parameter sets based on their posterior density:
istart=5000
npar=ParSet.shape[0]-istart
step=1#np.round(ParSet[istart:,].shape[0]/npar).astype('int')
idx=np.arange(istart,ParSet.shape[0],step)
X=ParSet[idx,0:MCMCPar.n]

# Run the CRN model again for these parameter sets
# (with a high-resolution depth profile to get a nice plot)
depth=np.arange(0,380,5)
pfx=np.zeros((npar,len(depth)))

from mcmc_func import forward_model_crn as forward_process
extra_par=[]
extra_par.append([Extra.P_spallation])
extra_par.append([depth])
extra_par.append([Extra.rho])
extra_par.append([Extra.att_length_spallation])
extra_par.append([Extra.decay_const])
extra_par.append([Extra.P_neg_muon_capture])
extra_par.append([Extra.att_length_neg_muon_capture])
extra_par.append([Extra.P_fast_muon_reactions])
extra_par.append([Extra.att_length_fast_muon_reactions]) 
for qq in xrange(0,npar):
    pfx[qq,:]=forward_process(X[qq,:].reshape((1,X.shape[1])),len(depth),extra_par)

# And run MAP solution:
ii=np.where(ParSet[:,-1]==np.max(ParSet[:,-1]))
ii=ii[0][0]
mapfx=forward_process(ParSet[ii,:MCMCPar.n].reshape((1,MCMCPar.n)),len(depth),extra_par)
# Compute 95\% interval around each simulated value
pfxs=np.sort(pfx,axis=0) 

upl=np.round(0.975*npar).astype('int')
lowl=np.round(0.025*npar).astype('int')
y1=pfxs[lowl-1,:]
y2=pfxs[upl-1,:]


#%% PLot predictive uncertainty
filename='PredUncMCMC'
savefigure=False
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
h1=ax.fill_between(np.concatenate((y1,y2[::-1])),np.concatenate((depth,depth[::-1])),color=np.array([0.8,0.8,0.8]),label='95% Uncertainty')
plt.hold(True)
h2,=plt.plot(mapfx,depth,color='black',label='MAP solution',linewidth=1.2)
plt.hold(True)
h3,=plt.plot(Measurement.MeasData[0,:],Extra.depth, "+", markeredgewidth=1.5, markersize=10,markeredgecolor='r',markerfacecolor='none',label='Measurements')
plt.gca().invert_yaxis()
plt.ylim(375, 0)
plt.xlim(50000, 200000)
xlabel=r'$\rm \ In \ situ \ ^{10}Be \ [atoms/g]$'
ylabel=r'$\rm \ Depth \ [cm]$'
ax.set_ylabel(ylabel, fontsize=22)
ax.set_xlabel(xlabel, fontsize=22)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(handles=[h1, h2,h3],fontsize=20,loc=4,frameon=False)
plt.show()
if savefigure==True:
    fig.savefig(filename+'.png',dpi=600)

