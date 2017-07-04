# -*- coding: utf-8 -*-
"""
A Python 2.7 implementation of grid-based Monte Carlo (MC) sampling for the Bayesian 
inversion of a CRN depth profile considered in Laloy et al. (2017). This MC sampling 
follows the procedure described in Marrero et al. (2016). Also, this "run_mc" script is 
separated in "#%%" sections that can conveniently be run separately with Spyder.

@author: Eric Laloy <elaloy@sckcen.be>, January 2017.

License: MIT.

Please drop me an email if you have any question and/or if you find a bug in this
program. 

Also, if you find this code useful please cite the paper for which it has been 
developed (Laloy et al., 2017).

References:
    
Laloy, E., Beertem, K., Vanacker, V., Christl, M., Rogiers, B., Wauters, L., 
    Bayesian inversion of a CRN depth profile to infer Quaternary erosion of the 
    northwestern Campine Plateau (NE Belgium), Earth Surf. Dynam., 5, 331–345, 
    2017, https://doi.org/10.5194/esurf-5-331-2017.
    
Marrero, S.M., Phillips, F.M., Borchers, B., Lifton, N., Aumer, R., Balco, G., 
    Cosmogenic nuclide systematics and the CRONUScalc program, Quat. Geochronol. 
    31, 199-219, 2016.    
                                                                                               
"""
import os
main_dir=r'D:\CRN_ProbabilisticInversion\MC_Inversion' # Set working directory
os.chdir(main_dir)
import time
import numpy as np
import mc
#% Set rng_seed
rng_seed=1 # np.random.seed(np.floor(time.time()).astype('int'))

# Other settings
npar=3 # Erosion rate, Age, Ninh
#Bounds
lb=np.array([2.0,0.,1e4]).reshape(1,npar)
ub=np.array([60.0,1e6,9e4]).reshape(1,npar)
Prior=Prior='Prior_CRN_1' 
#'Prior_CRN_1': Gaussian prior for erosion rate, uniform priors for Age and Ninh
#'Uniform': uniform prior for every parameter

sampling_strategy='Lattice' # 'Lattice' or 'LHS' (latin hypercube sampling)
ndiv=60 # Number of divisions per dimension in case of Lattice
         # Total number of samples is calculated by ndiv**npar
steps= 10#ndiv# Number of sampling rounds (only useful to store intermdiate results)

meas_filename='CRN_data.txt'

if __name__ == '__main__':
    
    start_time = time.time()
    
    q=mc.Sampler(n=npar,Prior=Prior,sampling_strategy=sampling_strategy,
                 ndiv=ndiv,DoParallel=False,parallel_jobs=8,rng_seed=rng_seed,
                 lb=lb,ub=ub,meas_filename=meas_filename,steps=steps)
    
    print("Iterating")
    
    tmpFilePath=None # None or main_dir+'\out_tmp.pkl'
    
    Xi, prior, of, lik, log_lik, fx, MCPar, MCVar, Extra, Measurement  = q.sample(RestartFilePath=tmpFilePath)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
    
    
#%%
#import os    
#main_dir=r'D:\CRN_ProbabilisticInversion\MC_Inversion'
#os.chdir(main_dir)
#import numpy as np 
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)  

with open('mc_out'+'.pkl','rb') as f:
    tmp_obj=pickle.load(f)
try:
    Xi=tmp_obj['Xi']
    of=tmp_obj['of']
    lik=tmp_obj['lik']
    log_lik=tmp_obj['log_lik']
    prior=tmp_obj['prior']
    fx=tmp_obj['fx']
    MCPar=tmp_obj['MCPar']
    Measurement=tmp_obj['Measurement']
    Extra=tmp_obj['Extra']
    MCMCVar=tmp_obj['MCVar']
    Modelname=tmp_obj['ModelName']
    print(MCVar.Iter)
except:
    pass
del tmp_obj

RecompLik=False
if RecompLik==True: # Recompute Gaussian likelihood using a different sigma_e that is consistent with the actual data misfit
    sigma_e=10000
    try:
        del lik, log_lik
    except:
        pass
    lik=np.zeros((Xi.shape[0],1))
    log_lik=np.zeros((Xi.shape[0],1))
    for ii in xrange(Xi.shape[0]):
        SSR = (of[ii,0]**2)*Measurement.N# of is RMSE
        log_lik[ii,0]= - ( Measurement.N / 2.0) * np.log(2.0 * np.pi) - Measurement.N * np.log( sigma_e ) - 0.5 * np.power(sigma_e,-2.0) * SSR
        lik[ii,0]=(1.0/np.sqrt(2*np.pi* sigma_e**2))**Measurement.N * np.exp(- 0.5 * np.power(sigma_e,-2.0) * SSR)
#%% Check number of no-runs because the E x t limits are exceeded:
vv=Xi[:,0]*Xi[:,1]*1e-6 
qq=np.where((vv<1) | (vv>35))  
print('The effective number of model runs is: ',Xi.shape[0]-qq[0].size) 
#%% Set zero likelihood to those runs for which E x t is out of bounds
lik[qq,0]=0
log_lik[qq,0]=-1e300
   
#%%        
# Turn log-likelihood into a rescaled likelihood (rlik) 
# Rescaling is better because of numerical underflow issues with very small likelhioods
# (yet here this only solves the numerical underflow for a rather tiny fraction of the zero-likelihood parameter sets)
ii=np.where(log_lik==np.max(log_lik));print(ii)
ii=np.where(lik==np.max(lik));print(ii)
rlik=np.exp(log_lik-np.max(log_lik))
ii=np.where(rlik==np.max(rlik));print(ii)

# Reshape prior and rlik for marginalization
prior3=np.zeros((MCPar.ndiv,MCPar.ndiv,MCPar.ndiv))
rlik3=np.zeros((MCPar.ndiv,MCPar.ndiv,MCPar.ndiv))
teller=0
for ii in xrange(0,MCPar.ndiv): 
    for jj in xrange(0,MCPar.ndiv):
        for kk in xrange(0,MCPar.ndiv):
            prior3[ii,jj,kk]=prior[teller,0]
            rlik3[ii,jj,kk]=rlik[teller,0]
            teller=teller+1
            
# Marginalize using trapzoidal integration rule
# Calculate the posterior distribution using trapzoidal integration rule to 
# estimate the evidence
Eros_rate=MCPar.pardiv[0]
Age=MCPar.pardiv[1]
Ninh=MCPar.pardiv[2]
evid=np.trapz(np.trapz(np.trapz(rlik3*prior3,Ninh,axis=2),Age,axis=1),Eros_rate,axis=0)
posterior3=rlik3*prior3/evid
posterior1=posterior3.flatten()
ii=np.where((log_lik+np.log(prior))==np.max((log_lik+np.log(prior))));print(ii)
ii=np.where(posterior1==np.max(posterior1));print(ii)
print(of[ii,0])
#%%
# Marginalize posterior for Erosion rate
posterior_Eros=np.trapz(np.trapz(posterior3,Ninh,axis=2),Age,axis=1)
# Check:
print(np.trapz(posterior_Eros,Eros_rate))
print(np.sum(posterior_Eros*(Eros_rate[1]-Eros_rate[0])))

# Marginalize posterior for Age
posterior_Age=np.trapz(np.trapz(posterior3,Ninh,axis=2),Eros_rate,axis=0)
# Check:
print(np.trapz(posterior_Age,Age))
print(np.sum(posterior_Age*(Age[1]-Age[0])))

# Marginalize posterior for Ninh
posterior_Ninh=np.trapz(np.trapz(posterior3,Age,axis=1),Eros_rate,axis=0)
# Check:
print(np.trapz(posterior_Ninh,Ninh))
print(np.sum(posterior_Ninh*(Ninh[1]-Ninh[0])))

#%% Print MAP solutions
ii=np.where(posterior_Eros==np.max(posterior_Eros))
jj=np.where(posterior_Age==np.max(posterior_Age))
kk=np.where(posterior_Ninh==np.max(posterior_Ninh))
print(Eros_rate[ii], Age[jj], Ninh[kk])

#%% Marginal posterior plots
from scipy.stats import norm
from scipy import interpolate, stats
import matplotlib.gridspec as gridspec
sub_letter=['(a)','(b)','(c)']
  
filename='MarginalDistributionsMC'
savefigure=False
xlabel=[r'$E \rm \ [m/Myr]$',r'$t \rm \ [yr]$',r'${N}_{\rm inh}\rm \ [atoms/g]$',
        r'$\sigma_{\rm e}\rm \ [atoms/g]$']
ylabel=r'$\rm Density$'
xti=[np.array([0,20,40,60]),np.array([0,5e5,1e6]),np.array([10000,50000,90000]),
     np.array([5000,10000,15000,20000,25000])]

DoResampling=False # Interpolate (or not) the 50 posterior values onto a coarser 1D grid
                  # for better readability of the plots

savefigure=False

fig = plt.figure(figsize=(11,11))
gs = gridspec.GridSpec(2, 4)

for i in xrange(0,3):
    #Plot posteriors
    if i == 0:
        sub=plt.subplot(gs[0, 0:2])
        xvar=Eros_rate
        yvar=posterior_Eros
    elif i == 1:
        sub=plt.subplot(gs[0,2:])
        xvar=Age
        yvar=posterior_Age
    else:
         sub=plt.subplot(gs[1, 1:3])
         xvar=Ninh
         yvar=posterior_Ninh
    if DoResampling==True:
        # resample from 50 bins to nbins
        nbins=25
        bins = np.linspace(MCPar.lb[0,i], MCPar.ub[0,i], nbins)
        f = interpolate.interp1d(xvar, yvar)
        yvar = f(bins)
        xvar=bins

    else:
        nbins=50 #Original number of bins
    #delta=np.diff(xvar)[0]
    bwidth=0.9*(MCPar.ub[0,i]-MCPar.lb[0,i])/nbins
    
    plt.bar(xvar,yvar,width=bwidth,align='center')
    plt.hold(True)
    
    # Plot priors
    xx=np.linspace(MCPar.lb[0,i]*0.9,MCPar.ub[0,i]*1.1,1e4)
    if i==0: # Normal prior
        yy=norm.pdf(xx,MCPar.pmu[i],MCPar.psd[i])
    elif i==1: # Normal prior
        yy=np.zeros(len(xx))++1/(MCPar.ub[0,i]-MCPar.lb[0,i])    
    elif i ==2: # Uniform prior
        yy=np.zeros(len(xx))++1/(MCPar.ub[0,i]-MCPar.lb[0,i])    
    plt.plot(xx,yy,'-r');plt.hold(False)  
    
    # Labels, limits and others
    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    sub.axes.set_xlabel(xlabel[i], fontsize = 16.0)
    sub.axes.get_xaxis().set_ticks(xti[i])   
    sub.axes.set_xlim(MCPar.lb[0,i], MCPar.ub[0,i])
    sub.axes.tick_params(labelsize=12)
    
    if (i==0 or i==2):
        sub.axes.set_ylabel(ylabel, fontsize=16)  
        sub.axes.set_ylabel(ylabel, fontsize = 16.0)
        sub.text(0.04, 0.92, sub_letter[i], transform=sub.axes.transAxes, 
            size=14)
    else:
        sub.text(0.90, 0.92, sub_letter[i], transform=sub.axes.transAxes, 
            size=14)
        
plt.tight_layout()
plt.show()

if savefigure==True:
    fig.savefig(filename+'.png',dpi=600)
    
#%% Predictive uncertainty plots
# Sample npar parameter sets based on their posterior density:
npar=50000
pk=posterior1/np.sum(posterior1)
xk=np.arange(0,posterior1.shape[0],1)
custm = stats.rv_discrete(name='custm', values=(xk, pk))
idx = custm.rvs(size=npar)
X=Xi[idx,:]

# Run the CRN model again for these parameter sets
# (with a high-resolution depth profile to get a nice plot)
depth=np.arange(0,380,5)
pfx=np.zeros((npar,len(depth)))
from mc_func import forward_model_crn as forward_process
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
ii=np.where(posterior1==np.max(posterior1))
mapfx=forward_process(Xi[ii,:].reshape((1,Xi.shape[1])),len(depth),extra_par)
# Compute 95\% interval around each simulated value
pfxs=np.sort(pfx,axis=0) 
upl=np.round(0.975*npar).astype('int')
lowl=np.round(0.025*npar).astype('int')
y1=pfxs[lowl-1,:]
y2=pfxs[upl-1,:]

# PLot predictive uncertainty
filename='PredUncMC'
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

