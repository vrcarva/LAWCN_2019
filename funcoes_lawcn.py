# -*- coding: utf-8 -*-
"""
Functions used in LAWN_epileptor.py
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
#import peakutils
from scipy import signal
import matplotlib.pyplot as plt

#%% EPILEPTOR
def eulerEpileptor(y, n, u, h, P ): #Model ODEs
    #y = current state, n = number of states, x = current time sample, h = timestep, P = parameters
    #y = [0:x1 1:y1 2:z 3:x2 4:y2 5:g]
    
    dydx = np.zeros(n)
    sigmaNoise = np.array([0.025,0.025,0.0,0.25,0.25,0.])*0.01
    #sigmaNoise = np.zeros(n)

    dydx[0] = y[1] - y[2] +  P["Iext1"] - f1(y[0],y[3],y[2])
    dydx[1] = P["y0"] - 5*(y[0]**2) - y[1]
    dydx[2] = (4*(y[0]-P["x0"])-y[2])/P["tau0"]
    dydx[3] = -y[4] + y[3] - y[3]**3 + P["Iext2"] + 2*y[5] - 0.3*(y[2] - 3.5)
    dydx[4] = (-y[4]+ f2(y[3]))/P["tau2"]
    dydx[5] = -P["gamma"]*(y[5]-0.1*y[0])
    
    yout = np.empty((0))
    for i in range(n):
        yout = np.append(yout,y[i] + 0.01*u[i] + h *dydx[i]+ dW(h)*np.sqrt(sigmaNoise[i])) 
    return yout


#define functions
def f1(x1,x2,z):
    if x1 <0:
        return x1**3 - 3.*(x1**2)
    else:
         return (x2 - 0.6*((z - 4)**2))*x1 
     
def f2(x2):
    if x2 < -0.25:
        return 0.
    else:
        return 6.*(x2+0.25)
    
def dW(delta_t):
    return np.random.normal(loc = 0.0, scale = np.sqrt(delta_t))
    

#%%
def fun_extractERPfeatsUni(erp,preERP,Fs):
    #Extracts univariate features from erp signal
    #preERP is the baseline signal
    #Fs is sampling frequency, in Hz

    featsOut = dict()
    #FEATURE EXTRACTION
    #Normalized energy: postStim/preStim
    featsOut["normEnergy"] = np.mean(erp**2)/np.mean(preERP**2) #normalized energy 
    featsOut["Energy"] = np.mean(erp**2) 
    # Statistical Moments
    featsOut["Var"] =  np.var(erp)
    featsOut["Skew"] =  skew(erp)
    featsOut["Kurt"] =  kurtosis(erp)
    dyTrecho = np.diff(erp)
    dyBasal = np.diff(preERP)
    #hmobBasal = np.sqrt(np.var(dyBasal)/np.var(simulatedLFP[mi,stimTS[si]-int(tprePEARP*Fs):stimTS[si]]))
    #hcompBasal =  np.sqrt(np.var(np.diff(dyBasal))/np.var(dyBasal))/hmobBasal
    featsOut["Hmob"] = np.sqrt(np.var(dyTrecho)/featsOut["Var"])
    featsOut["Hcomp"] =  np.sqrt(np.var(np.diff(dyTrecho))/np.var(dyTrecho))/featsOut["Hmob"]
    featsOut["PkAmp"] = max(erp)#maximum ERP value
    featsOut["PkLag"] = np.argmax(erp)/Fs 
    erpVale = erp[np.argmax(erp):]#minimum after the initial peak
    featsOut["ValeAmp"] = min(erpVale)
    featsOut["ValeLag"] = np.argmin(erpVale)/Fs
    #lag 1 autocorrelation
    featsOut["lag1AC"]= np.corrcoef(erp[:-1], erp[1:])[0,1]
     
    return featsOut

def fun_extractERPfeatsMultivar(erpSynch,erpSynchFilt,Fs):
    #fun_extractERPfeatsMultivar(erpSynch,erpSynchFilt,Fs):
    #erpSynch is a lxN matrix --> L channels with N samples each, from which synchrony measures are taken (PLV and correlation)
    #detrend and normalize erpSynch (fucks up PLV values in some cases if it's not detrended)
    #erpSynchFilt is the filtered version of erpSynch, for calculating the PLV
    from itertools import combinations
    erpSynch = (erpSynch.T - np.mean(erpSynch,axis = 1)).T
    erpSynch = (erpSynch.T/np.std(erpSynch,axis = 1)).T
    erpSynchFilt = (erpSynchFilt.T - np.mean(erpSynchFilt,axis = 1)).T
    featsOut = dict()
    #for all channel pair combinations
    featsOut["combinations"] = list(combinations(range(erpSynch.shape[0]), 2))
    #featsOut["Corr"] = np.zeros(len(featsOut["combinations"]),erpSynch.shape[1])
    #featsOut["PLV"] = np.zeros(len(featsOut["combinations"]),erpSynch.shape[1])
    #*** Coupling Measures ***
    #correlation  (max value)
    featsOut["Corr"] = [np.max(signal.correlate(erpSynch[ki[0],:],
             erpSynch[ki[1],:]))/erpSynch.shape[1] for ki in featsOut["combinations"]]
    
    featsOut["CorrCoefs"] = np.corrcoef(erpSynch)

    #PLV 
    phases = np.array([np.angle(signal.hilbert(erpSynchFilt[ki,:])) for ki in range(erpSynchFilt.shape[0])])
    featsOut["PLV"] = [np.abs(np.sum(np.exp(1j*(phases[ki[0]]-phases[ki[1]])))/
            phases.shape[1]) for ki in featsOut["combinations"]]
    featsOut["PLVphase"] =  [np.mean(np.unwrap(phases[ki[0]]-phases[ki[1]],axis = 0)) for ki in featsOut["combinations"]]
    
    #Coherence
    featsOut["Coh"] = np.zeros([len(featsOut["combinations"])])
    iind = 0
    for ki in featsOut["combinations"]:
        Wxy, Cxy = signal.coherence(erpSynch[ki[0],:], erpSynch[ki[1],:], Fs, noverlap = 200)
        featsOut["Coh"][iind] = np.mean(Cxy[0:11])#0-40 Hz
        iind+=1
    
    
    return featsOut


    
#%
def LAWCN_figFeat(feats,tFeats,yline,ytext):
    plt.figure()
    ax1 = plt.subplot(321)
    ax1.plot(tFeats,feats["Var"],color = 'k')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.locator_params(axis='y', nbins=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False) 
    ax1.get_xaxis().set_ticks([])
    plt.title("Variance")
    
    ax2 = plt.subplot(322)
    ax2.plot(tFeats,feats["Skew"],color = 'k')
    ax2.locator_params(axis='y', nbins=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False) 
    ax2.get_xaxis().set_ticks([])    
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title("Skewness")
    
    ax3 = plt.subplot(323)
    ax3.plot(tFeats,feats["Kurt"],color = 'k')
    ax3.locator_params(axis='y', nbins=2)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False) 
    ax3.get_xaxis().set_ticks([])
    ax3 = plt.title("Kurtosis")
    
    ax4 = plt.subplot(324)
    p4x = ax4.plot(tFeats,feats["Hmob"],color = 'k')
    ax4.set_ylabel('Mobility',color = 'k') 
    ax4.locator_params(axis='y', nbins=2)
    ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.spines['top'].set_visible(False)
    #ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False) 
    ax4.get_xaxis().set_ticks([])    
    
    ax4a = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    ax4a.set_ylabel('Complexity',color = 'blue') 
    p4ax = ax4a.plot(tFeats,feats["Hcomp"],color = 'blue')
    ax4a.locator_params(axis='y', nbins=2)
    ax4a.autoscale(enable=True, axis='x', tight=True)
    ax4a.spines['top'].set_visible(False)
    #ax4a.spines['right'].set_visible(False)
    ax4a.spines['bottom'].set_visible(False) 
    ax4a.get_xaxis().set_ticks([])    
    plt.title("Hjorth Parameters")
    #ax4a.legend(p4x + p4ax,['Mobility','Complexity'], loc=0)

    ax5 = plt.subplot(325)
    ax5.plot(tFeats,feats["lag1AC"],color = 'k')
    ax5.locator_params(axis='y', nbins=2)
    ax5.autoscale(enable=True, axis='x', tight=True)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False) 
    ax5.get_xaxis().set_ticks([])    
    plt.title("Lag-1 autocorrelation")
    
    ax6 = plt.subplot(326)
    ax6.plot(tFeats,feats["PkAmp"],color = 'k')
    ax6.locator_params(axis='y', nbins=2)
    ax6.autoscale(enable=True, axis='x', tight=True)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_visible(False) 
    ax6.get_xaxis().set_ticks([])    
    ax6.plot([200,700],[yline,yline],linewidth = 3,color = 'k')
    plt.title("Peak amplitude")
    ax6.text(450,ytext, '500 s',  fontsize=9,horizontalalignment='center',verticalalignment='center')  
    