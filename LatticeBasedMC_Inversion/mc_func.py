# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>, January 2017.
"""
import numpy as np
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import sys


def lhs(minn,maxn,N): # Latin Hypercube sampling
    # Here minn and maxn are assumed to be 1xd arrays 
    x = np.zeros((N,minn.shape[1]))

    for j in xrange (0,minn.shape[1]):
    
        idx = np.random.permutation(N)
        P =(idx - x[:,j])/N
        x[:,j] = minn[0,j] + P*(maxn[0,j] - minn[0,j])

    return x
    
def CompLikelihood(X,fx,MCPar,Measurement,Extra):
    
    Sigma=Measurement.Sigma*np.ones((X.shape[0]))
    of=np.zeros((fx.shape[0],1))
    p=np.zeros((fx.shape[0],1))
    log_p=np.zeros((fx.shape[0],1))
    for ii in xrange(0,fx.shape[0]):
        e=Measurement.MeasData-fx[ii,:]
       
        of[ii,0]=np.sqrt(np.sum(np.power(e,2.0))/e.shape[1])
        if MCPar.lik==2: # Compute standard uncorrelated and homoscedastic Gaussian log-likelihood
            log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(2.0 * np.pi) - Measurement.N * np.log( Sigma[ii] ) - 0.5 * np.power(Sigma[ii],-2.0) * np.sum( np.power(e,2.0) )
            p[ii,0]=(1.0/np.sqrt(2*np.pi* Sigma[ii]**2))**Measurement.N * np.exp(- 0.5 * np.power(Sigma[ii],-2.0) * np.sum( np.power(e,2.0) ))
            
        if MCPar.lik==3: # Box and Tiao (1973) log-likelihood formulation with Sigma integrated out based on prior of the form p(sigma) ~ 1/sigma
            log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(np.sum(np.power(e,2.0))) 
            p[ii,0]=np.exp(log_p[ii,0])
    return of, p, log_p
    
def CompPrior(X,MCPar):
    
    prior=np.ones((X.shape[0],1))
    log_prior=np.zeros((X.shape[0],1))
    for ii in xrange(0,X.shape[0]):
        if (MCPar.Prior[0:9]=='Prior_CRN'):
            # Uniform prior for Ninh
            prior[ii,0] = 1.0/(MCPar.ub[0,MCPar.idx_unif2]-MCPar.lb[0,MCPar.idx_unif2])
            log_prior[ii,0]=np.log(prior[ii,0])
            
            # Uniform prior for Age
            prior[ii,0] = 1.0/(MCPar.ub[0,MCPar.idx_unif1]-MCPar.lb[0,MCPar.idx_unif1])
            log_prior[ii,0]=np.log(prior[ii,0])
            
            if (MCPar.Prior[10]=='1'): # Gaussian prior for erosion rate
                prior[ii,0] = prior[ii,0]*norm.pdf(X[ii,MCPar.idx_norm],MCPar.pmu,MCPar.psd)
                log_prior[ii,0]+=np.log(prior[ii,0])
            else: # Uniform prior for erosion rate
                prior[ii,0] = prior[ii,0]*(1.0/(MCPar.ub[0,MCPar.idx_unif0]-MCPar.lb[0,MCPar.idx_unif0]))
                log_prior[ii,0]+=np.log(prior[ii,0])
            
            # Check log_p for inf
            if np.isinf(log_prior[ii,0])==True:
                log_prior[ii,0]=1e-200
        else: # Uniform prior for every variable
            for jj in xrange(0,MCPar.n):
                prior[ii,0] = prior[ii,0]*(1.0/(MCPar.ub[0,jj]-MCPar.lb[0,jj])) 
                log_prior[ii,0]+=np.log(prior[ii,0])
    return prior, log_prior
    
def forward_parallel(forward_process,X,n,n_jobs,extra_par): 
    
    n_row=X.shape[0]
    
    parallelizer = Parallel(n_jobs=n_jobs)
    
    tasks_iterator = ( delayed(forward_process)(X_row,n,extra_par) 
                      for X_row in np.split(X,n_row))
         
    result = parallelizer( tasks_iterator )
    # Merging the output of the jobs
    return np.vstack(result)
    
      
def RunFoward(X,MCPar,Measurement,ModelName,Extra,DNN=None):
    
    n=Measurement.N
    n_jobs=Extra.n_jobs
        
    if ModelName=='forward_model_crn':
        extra_par=[]
        extra_par.append([Extra.P_spallation])
        extra_par.append([Extra.depth])
        extra_par.append([Extra.rho])
        extra_par.append([Extra.att_length_spallation])
        extra_par.append([Extra.decay_const])
        extra_par.append([Extra.P_neg_muon_capture])
        extra_par.append([Extra.att_length_neg_muon_capture])
        extra_par.append([Extra.P_fast_muon_reactions])
        extra_par.append([Extra.att_length_fast_muon_reactions]) 
        
    else:
        extra_par=None
    
    forward_process=getattr(sys.modules[__name__], ModelName)
    
    if MCPar.DoParallel==True:
    
        start_time = time.time()
        
        fx=forward_parallel(forward_process,X,n,n_jobs,extra_par)

        end_time = time.time()
        elapsed_time = end_time - start_time
    
        #print("Parallel forward calls done in %5.4f seconds." % (elapsed_time))
    else:
        fx=np.zeros((X.shape[0],n))
        
        start_time = time.time()
        
       # X needs to be a 1-dim array instead of a vector
        for qq in xrange(0,X.shape[0]):
            fx[qq,:]=forward_process(X[qq,:].reshape((1,X.shape[1])),n,extra_par)
      
        end_time = time.time()
        elapsed_time = end_time - start_time
         
        #print("Sequential forward calls done in %5.4f seconds." % (elapsed_time))
            
    return fx
    
def forward_model_crn(X,n,par):
    
    # X is a 1-d array and not a vector
    Texp=X[0,1]
    Eros=X[0,0]*1e-4 # From m/Myr to cm/yr
    Ninh=X[0,2]
    P_spallation=par[0][0]
    depth=par[1][0]
    rho=par[2][0]
    att_length_spallation=par[3][0]
    decay_const=par[4][0]
    P_neg_muon_capture=par[5][0]
    att_length_neg_muon_capture=par[6][0]
    P_fast_muon_reactions=par[7][0]
    att_length_fast_muon_reactions=par[8][0]
    
    term1=(((P_spallation*np.exp(-depth*rho/att_length_spallation))/
    (decay_const+rho*Eros/att_length_spallation))*(1-np.exp(-Texp*
    (decay_const + rho*Eros/att_length_spallation))))
    
    term2=(((P_neg_muon_capture*np.exp(-depth*rho/att_length_neg_muon_capture))/
    (decay_const + rho*Eros/att_length_neg_muon_capture))*
    (1-np.exp(-Texp*(decay_const + rho*Eros/att_length_neg_muon_capture))))
    
    term3=(((P_fast_muon_reactions*np.exp(-depth*rho/att_length_fast_muon_reactions))/
    (decay_const + rho*Eros/att_length_fast_muon_reactions))*
    (1-np.exp(-Texp*(decay_const + rho*Eros/att_length_fast_muon_reactions))))
    
    out=Ninh + term1 + term2 + term3

    return out


