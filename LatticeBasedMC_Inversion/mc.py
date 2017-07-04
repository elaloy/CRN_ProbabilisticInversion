# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>, January 2017.
"""
from __future__ import print_function
#import time
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import time

from mc_func import* 

from attrdict import AttrDict


MCPar=AttrDict()

MCVar=AttrDict()

Measurement=AttrDict()

Extra=AttrDict()

class Sampler:
    
    def __init__(self, lb,ub,meas_filename,steps,n=3, Prior='Prior_CRN_1',sampling_strategy='Lattice',
                 ndiv=100,DoParallel=False,parallel_jobs=None,rng_seed=1,savemodout=True,
                 saveout=True,save_tmp_out=False,ModelName='forward_model_crn'):
     
        MCPar.n=n
        MCPar.Prior=Prior
        MCPar.sampling_strategy=sampling_strategy
        MCPar.ndiv=ndiv
        MCPar.DoParallel=DoParallel
        MCPar.parallel_jobs=parallel_jobs
        MCPar.savemodout=savemodout
        MCPar.saveout=saveout
        MCPar.save_tmp_out=save_tmp_out
        np.random.seed(rng_seed)
        MCPar.rng_seed=rng_seed
        ModelName=ModelName
        MCPar.lb=lb
        MCPar.ub=ub
        MCPar.steps=steps
        MCPar.ndraw=MCPar.ndiv**MCPar.n
        MCPar.gen_size=MCPar.ndraw/MCPar.steps # ndraw/steps must be a round number
        Extra.n_jobs=parallel_jobs
        
        if (MCPar.Prior[0:9]=='Prior_CRN'):
            
            if (MCPar.Prior[10]=='1'): # Gaussian prior for ersion rate
                MCPar.idx_norm=np.array([0])
                MCPar.psd=0.5*(MCPar.ub[0,MCPar.idx_norm]-MCPar.lb[0,MCPar.idx_norm])
                MCPar.pcov=np.eye(len(MCPar.idx_norm))*(MCPar.psd**2)
                MCPar.invC=np.linalg.inv(MCPar.pcov)
                MCPar.pmu=MCPar.lb[0,MCPar.idx_norm]+0.5*(MCPar.ub[0,MCPar.idx_norm]-MCPar.lb[0,MCPar.idx_norm])
            else:
                MCPar.idx_unif0=0 # Uniform prior for erosion rate
                
            # Uniform prior for Ninh
            MCPar.idx_unif2=2
            # Uniform prior for Age
            MCPar.idx_unif1=1
            
        # Load measurements
        raw_data=np.loadtxt(meas_filename)
        # Measusrements 4 (155 cm depth) and 6 (235 cm depth) are unreliable 
        # and were thus removed
        Measurement.idx=np.concatenate((np.arange(0,4),np.array([5,7,8])))
        Measurement.MeasData=raw_data[Measurement.idx,1] # Col 1 contains the measured concentrations
        Measurement.N=len(Measurement.MeasData)
        Measurement.MeasData=Measurement.MeasData.reshape(1,Measurement.N) # Make a 1 x N array to avoid later problems with likelihood calculations
        Extra.depth=raw_data[Measurement.idx,0]# Col 0 contains the measurement depths
        meas_err=raw_data[Measurement.idx,2]# Col 2 contains the standard deviations of the measurement errors

        # For the inversion we take a single standard deviation of the measurement
        # errors equal to the maximum one. This is just fine as the model cannot
        # simulate the measurement data within this maximum measurement error anyway            
        Measurement.Sigma=np.max(meas_err)
        del raw_data, meas_err
        
        # Set various (fixed) CRN model parameters:
        Extra.P_spallation= 4.449#atoms/g/yr
        Extra.rho=1.7 #g/cm3
        Extra.att_length_spallation=152 #g/cm2
        Extra.decay_const=4.997e-7 #yr-1
        Extra.P_neg_muon_capture=0.030 #atoms/g/yr
        Extra.att_length_neg_muon_capture=1500 #g/cm2
        Extra.P_fast_muon_reactions=0.027 #atoms/g/yr
        Extra.att_length_fast_muon_reactions=4320 #g/cm2       
    
        # Type of Likelihood function
        MCPar.lik=2
        
        self.MCPar=MCPar
        self.Measurement=Measurement
        self.Extra=Extra
        self.ModelName=ModelName
       
    def _init_sampling(self):
        
        if self.MCPar.sampling_strategy=='Lattice':
            pardiv=[]
            for ii in xrange(0,self.MCPar.n):
                pardiv.append(np.linspace(self.MCPar.lb[0,ii],self.MCPar.ub[0,ii],self.MCPar.ndiv))
            self.MCPar.pardiv=pardiv;del pardiv
            
            Xi=np.zeros((self.MCPar.ndraw,self.MCPar.n))
            teller=0
            for ii in xrange(0,self.MCPar.ndiv): 
                for jj in xrange(0,self.MCPar.ndiv):
                    for kk in xrange(0,self.MCPar.ndiv):
                        Xi[teller,0]=self.MCPar.pardiv[0][ii]
                        Xi[teller,1]=self.MCPar.pardiv[1][jj]
                        Xi[teller,2]=self.MCPar.pardiv[2][kk]
                        teller=teller+1
                print('Building the sample matrix : ' +str(teller) + ' samples done')        

        else:
            if (self.MCPar.Prior[0:9]=='Prior_CRN'): 
                # First draw samples from uniform distribution for all variables and replace as needed after
                Xi=lhs(self.MCPar.lb,self.MCPar.ub,self.ndraw) 
                if (self.MCMCPar.Prior[10]=='1'): # Gaussian prior for Eros
                    X[:,self.MCPar.idx_norm]=np.random.multivariate_normal(self.MCPar.pmu,self.MCPar.pcov,self.MCPar.ndraw) # Replace by normal distribution for the 0:MCMCPar.ngp variables
           
            else: # Uniform prior, LHS sampling
                Xi=lhs(self.MCPar.lb,self.MCPar.ub,self.MCPar.ndraw)
                
        of=np.zeros((self.MCPar.ndraw,1))
        lik=np.zeros((self.MCPar.ndraw,1))
        prior=np.zeros((self.MCPar.ndraw,1))
        log_lik=np.zeros((self.MCPar.ndraw,1))
        
        self.of=of
        self.lik=lik
        self.prior=prior
        self.log_lik=log_lik
        self.Xi=Xi
        
        if not(self.MCPar.savemodout==True):
            self.fx=None
        
        MCVar.Iter=0
        self.MCVar=MCVar
        
    def sample(self,RestartFilePath=None):
        
        if not(RestartFilePath is None):
            print('This is a restart')
            with open(RestartFilePath, 'rb') as fin:
                tmp_obj = pickle.load(fin)
            self.Xi=tmp_obj['Xi']
            self.of=tmp_obj['of']
            self.lik=tmp_obj['lik']
            self.log_lik=tmp_obj['log_lik']
            self.prior=tmp_obj['prior']
            self.fx=tmp_obj['fx']
            self.MCPar=tmp_obj['MCPar']
            self.MCVar=tmp_obj['MCVar']
            self.Measurement=tmp_obj['Measurement']
            self.ModelName=tmp_obj['ModelName']
            self.Extra=tmp_obj['Extra']
            del tmp_obj
            # Reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
        else:
            self._init_sampling()
        
       
        # Main sampling loop  
        while self.MCVar.Iter < self.MCPar.ndraw:
            
            # Loop a number of times for intermediate saving if wished
            for gen_number in xrange(0,self.MCPar.steps):
                
                xnew=self.Xi[self.MCVar.Iter:(self.MCVar.Iter+self.MCPar.gen_size),:]
                
                fx_new = RunFoward(xnew,self.MCPar,self.Measurement,self.ModelName,self.Extra)
                 
                # Compute the likelihood of each proposal in each chain
                of_new, lik_new, log_lik_new = CompLikelihood(xnew,fx_new,self.MCPar,self.Measurement,self.Extra)
    
                # Compute prior
                prior_new, log_prior_new = CompPrior(xnew,self.MCPar)
                               
                # Store results
                self.of[self.MCVar.Iter:(self.MCVar.Iter+self.MCPar.gen_size),:]=of_new; del of_new
                self.lik[self.MCVar.Iter:(self.MCVar.Iter+self.MCPar.gen_size),:]=lik_new; del lik_new
                self.log_lik[self.MCVar.Iter:(self.MCVar.Iter+self.MCPar.gen_size),:]=log_lik_new; del log_lik_new
                self.prior[self.MCVar.Iter:(self.MCVar.Iter+self.MCPar.gen_size),:]=prior_new; del prior_new
                
                
               # Check whether to store the simulation results of the function evaluations
                if self.MCPar.savemodout==True:
                    if self.MCVar.Iter==0:
                        self.fx=fx_new
                    else:
                        self.fx=np.append(self.fx,fx_new,axis=0)
                    del fx_new
                
                # Update Iter
                self.MCVar.Iter = self.MCVar.Iter + self.MCPar.gen_size
                print('Iter = ',self.MCVar.Iter)

                if self.MCPar.save_tmp_out==True:
                    with open('out_tmp'+'.pkl','wb') as f:
                        pickle.dump({'Xi':self.Xi,'of':self.of,
                        'lik':self.lik,'prior':self.prior,'log_lik':self.log_lik,'fx':self.fx,'MCPar':self.MCPar,
                        'MCVar':self.MCVar,'Measurement':self.Measurement,
                        'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.MCPar.saveout==True:
            with open('mc_out'+'.pkl','wb') as f:
                pickle.dump({'Xi':self.Xi,'of':self.of,
                        'lik':self.lik,'prior':self.prior,'log_lik':self.log_lik,'fx':self.fx,'MCPar':self.MCPar,
                        'MCVar':self.MCVar,'Measurement':self.Measurement,
                        'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.Xi, self.prior, self.of, self.lik, self.log_lik,  self.fx, self.MCPar, self.MCVar,self.Extra, self.Measurement      