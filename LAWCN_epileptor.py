# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:54:30 2018

@author: Vinícius Rezende Carvalho
Simulates EPILEPTOR model (with or without periodic stimulation) and extract features periodically 
(one segment for each stimuli)

"""
import numpy as np
import matplotlib.pyplot as plt
from funcoes_lawcn import fun_extractERPfeatsUni, fun_extractERPfeatsMultivar,eulerEpileptor,LAWCN_figFeat
from scipy import signal
import scipy.fftpack


#%% Simulation Parameters

nODEs = 6 # Number of ODEs
finalTime = 8300 #simulation time, in seconds
Fs = 512. #user-defined (typical: 512 Hz)
#Stimulus 
stimDuration = 0.02 #Stimuli duration (seconds)
stimPeriod = 2 #Inter-stimuli interval, in seconds
tprePEARP = 0.4 #pre-stimuli epoch (seconds)
tposPEARP = 0.4#post-stimuli epoch (seconds)
stimAmp = 0. #stimuli amplitude

promedNstim = 20 #Number of samples to average features (or ERPs - Evoked response potentials)
promedOverlap = 15 #Overlap of samples to average features (or ERPs)
P = {"y0":1.,"x0":-1.6,"tau0":2857.,"tau2":10.,"gamma":0.01,"Iext1":3.1,"Iext2":0.45} # model parameters
yold = np.array([0.,-5.,5.5,0.,0.,0.]) #initial conditions
promedia = "erp" #"features" or "erp" - takes sliding window mean from ERPs or from features?    
filtraHP = 1 #if 1, filters output signal
#design filters
b, a = signal.butter(3, 0.05*2/Fs, 'high')#highpass filter - 0.05 for control, 1.5 for stimuli
bLP, aLP = signal.butter(3, 20*2/Fs) #lowpass filter (only for synchrony measures)

dt = 1./Fs # time step or sampling period
nbSamples = int(finalTime / dt) # number of samples
x0Series = np.linspace(-4.,-2.,nbSamples)#Changes x0 over time or keep it constant?
#%


trajectory = np.zeros([3,nbSamples]) #3D Trajectory
simulatedLFP = np.zeros(nbSamples)#simlated LFP signal (x1 + x2)
tvec = np.zeros(nbSamples)#time vector
uinput = np.zeros([nODEs,nbSamples])#input variable
stimTS = np.arange(int(Fs*stimPeriod),nbSamples,int(Fs)*stimPeriod)#stimuli timestamps
stimIndxs = [np.arange(int(Fs*stimPeriod)+i,nbSamples-i,int(Fs)*stimPeriod) for i in range(int(Fs*stimDuration))]
stimIndxsNegative = [np.arange(int(Fs*2)+int(stimDuration*Fs)+i,nbSamples-i,int(Fs)*stimPeriod) for i in range(int(Fs*stimDuration))]
#assign stimuli series to x1 and y1
for i in [0,3]:
    uinput[i,stimIndxs] = stimAmp
    uinput[i,stimIndxsNegative] = -stimAmp#pulso bifásico  

t = 0.
xstates = np.zeros([nODEs,nbSamples])
nstim = len(stimTS)
promedIniidxs = np.arange(0,nstim-promedNstim,promedNstim-promedOverlap)

#SIMULATION
for tt in range(nbSamples):
    P["x0"] = x0Series[tt]
    ynew = eulerEpileptor(yold,nODEs,uinput[:,tt],dt,P)
    yold = ynew
    xstates[:,tt] = ynew
    #time vector, recorded LFP and states
    tvec[tt] = t
    t += dt
    simulatedLFP[tt] = -ynew[0] + ynew[3] #EPILEPTOR output
    trajectory[0,tt] = -ynew[0]
    trajectory[1,tt] = ynew[3]
    trajectory[2,tt] = ynew[2]

y = np.copy(simulatedLFP)
if filtraHP:
    simulatedLFP = signal.filtfilt(b, a, y)#filters output signal


#filters states (for PLV calculation)
xstatesFilt = np.zeros(xstates.shape)
for chi in range(nODEs):
    xstatesFilt[chi,:] =  signal.filtfilt(bLP, aLP, xstates[chi,:])
    xstatesFilt[chi,:] =  signal.filtfilt(b, a, xstatesFilt[chi,:])

#%% ERPs

Nmulti = len(promedIniidxs)
#univariate features
Feats = {"normEnergy":np.zeros(nstim),"Var":np.zeros(nstim),
         "Skew":np.zeros(nstim),"Kurt":np.zeros(nstim),
         "Hmob":np.zeros(nstim),"Hcomp":np.zeros(nstim),
         "PkAmp":np.zeros(nstim),"lag1AC":np.zeros(nstim),
         "ValeAmp":np.zeros(nstim),"PkLag":np.zeros(nstim),
         "ValeLag":np.zeros(nstim),"Energy":np.zeros(nstim)}
#synchrony features
FeatsMulti = {"PLVs":np.zeros([6,nstim]),"Corrs":np.zeros([6,nstim])}   
featLabels = Feats.keys() 
groupedFeatsPromed = dict([(fl,np.zeros(Nmulti)) for fl in featLabels ])
PLVsPromed = np.zeros([FeatsMulti["PLVs"].shape[0],Nmulti])
CorrsPromed = np.zeros([FeatsMulti["PLVs"].shape[0],Nmulti])
tFeatsPromed = np.zeros(Nmulti)
allPEARPS = np.zeros([nstim,int(tprePEARP*Fs)+int(tposPEARP*Fs)])

if promedia == "feature":#averages features
    print('feature averaging')
    for si in range(nstim):    #for each stimulus
        #ERP and features
        indsERP = np.arange(stimTS[si],stimTS[si]+int(tposPEARP*Fs))
        ERP = simulatedLFP[indsERP]
        preerp = simulatedLFP[stimTS[si]-int(tprePEARP*Fs):stimTS[si]]
        #detrend?
        #preerp = preerp - np.mean(preerp)
        #ERP = ERP - np.mean(ERP)
        allPEARPS[si,:] = simulatedLFP[stimTS[si]-int(tprePEARP*Fs):stimTS[si]+int(tposPEARP*Fs)]  
        allPEARPS[si,:] =  allPEARPS[si,:] - np.mean(preerp)#detrend
        featsTemp = fun_extractERPfeatsUni(ERP,preerp,Fs)
        featsTempMulti = fun_extractERPfeatsMultivar(xstates[np.ix_([0,1,3,4],indsERP)],xstatesFilt[np.ix_([0,1,3,4],indsERP)],Fs)
        
        for fKey in featsTemp.keys():
            groupedFeatsPromed[fKey][si] = featsTemp[fKey]
        #*** Coupling Measures ***
        FeatsMulti["Corrs"][:,si] = featsTempMulti["Corr"]
        FeatsMulti["PLVs"][:,si] = featsTempMulti["PLV"]
 
    groupedFeats = np.array([Feats[fl] for fl in Feats.keys()])

    #averages features
    for si in range(Nmulti):
        indxsTemp = range(promedIniidxs[si],promedIniidxs[si]+promedNstim)
        for ii,fkey in zip(range(len(featLabels)),featLabels):
            groupedFeatsPromed[fkey][si] = np.mean(groupedFeats[ii,indxsTemp])
        PLVsPromed[:,si] = np.mean(FeatsMulti["PLVs"][:,indxsTemp],axis = 1)
        CorrsPromed[:,si] = np.mean(FeatsMulti["Corrs"][:,indxsTemp],axis = 1)
        tFeatsPromed[si] = (stimTS[indxsTemp[(len(indxsTemp)+1)//2]])/Fs
else: #averages ERP(responses) before extracting features
    print('ERP averaging')

    statesAux = np.zeros([nstim,int(tposPEARP*Fs),4])
    for si in range(nstim):    
        #ERP and features
        indsERP = np.arange(stimTS[si],stimTS[si]+int(tposPEARP*Fs))
        ERP = simulatedLFP[indsERP]
        #ERP = ERP - np.mean(simulatedLFP[indsERP])
        preerp = simulatedLFP[stimTS[si]-int(tprePEARP*Fs):stimTS[si]]
        #preerp = preerp - np.mean(preerp)
        allPEARPS[si,:] = simulatedLFP[stimTS[si]-int(tprePEARP*Fs):stimTS[si]+int(tposPEARP*Fs)]    
        statesAux[si,:,:] = xstates[np.ix_([0,1,3,4],indsERP)].T
        
    for si in range(Nmulti):
        indxsTemp = range(promedIniidxs[si],promedIniidxs[si]+promedNstim)
        ErpPromed = np.mean(allPEARPS[indxsTemp,int(tprePEARP*Fs):],axis = 0)
        prePromed = np.mean(allPEARPS[indxsTemp,0:int(tprePEARP*Fs)-1],axis = 0)
        statesPromed = np.mean(statesAux[indxsTemp,:,:],axis = 0)
        tFeatsPromed[si] = (stimTS[indxsTemp[(len(indxsTemp)+1)//2]])/Fs
        
        featsTemp = fun_extractERPfeatsUni(ErpPromed,prePromed,Fs)
        featsTempMulti = fun_extractERPfeatsMultivar(statesPromed.T,statesPromed.T,Fs)
        
        for fKey in featsTemp.keys():
            groupedFeatsPromed[fKey][si] = featsTemp[fKey]       
        #*** Coupling Measures ***
        #correlation 
        CorrsPromed[:,si] = featsTempMulti["Corr"]
        PLVsPromed[:,si] = featsTempMulti["PLV"]

meanPEARPs = np.mean(allPEARPS,axis = 0)#mean evoked potential
stdPEARPs = np.std(allPEARPS,axis = 0)   
tpearp = np.linspace(0,meanPEARPs.shape[0]/Fs,meanPEARPs.shape[0])
#%% some figures
 
#plot simulated LFP
fig2 = plt.figure()
plt.plot(tvec,simulatedLFP,tvec,xstates[2,:])



fig4 = plt.figure()
plt.subplot(331)
plt.plot(tFeatsPromed,groupedFeatsPromed["Energy"])
plt.title("Energy")
plt.subplot(332)
plt.plot(tFeatsPromed,groupedFeatsPromed["normEnergy"])
plt.title("normEnergy")
plt.subplot(333)
plt.plot(tFeatsPromed,groupedFeatsPromed["Var"])
plt.title("Variance")
plt.subplot(334)
plt.plot(tFeatsPromed,groupedFeatsPromed["Skew"])
plt.title("Skewness")
plt.subplot(335)
plt.plot(tFeatsPromed,groupedFeatsPromed["Kurt"])
plt.title("Kurtosis")
plt.subplot(336)
plt.plot(tFeatsPromed,groupedFeatsPromed["Hmob"])
plt.title("Hjorth Mob")
plt.subplot(337)
plt.plot(tFeatsPromed,groupedFeatsPromed["Hcomp"])
plt.title("Hjorth Comp")
plt.subplot(338)
plt.plot(tFeatsPromed,groupedFeatsPromed["ValeLag"],tFeatsPromed,groupedFeatsPromed["PkLag"])
plt.title("Lags")
plt.legend(["Vale","Pico"])
plt.subplot(339)
plt.plot(tFeatsPromed,groupedFeatsPromed["ValeAmp"],tFeatsPromed,groupedFeatsPromed["PkAmp"])
plt.title("Amplitudes")
plt.legend(["Vale","Pico"])

#Synchronization
combtemp = [0,1,2,7]#quais pares plotar
#combtemp = range(10)
fig5 = plt.figure()
plt.subplot(121)
plt.plot(tFeatsPromed,CorrsPromed.T)
plt.title("Correlation")
plt.subplot(122)
plt.plot(tFeatsPromed,PLVsPromed.T)
plt.title("PLV")
plt.legend([ii for ii in featsTempMulti["combinations"]])



#%% plot EPILEPTOR simulation - figure 1 (stimAmp = 0)

fig1 = plt.figure()
grid1 = plt.GridSpec(1, 3)
ax1a = plt.subplot(grid1[0,0:2]) 
ax1a.plot(tvec,y,'k',linewidth = 0.8)
ax1a.autoscale(enable=True, axis='x', tight=True)
ax1a.spines['top'].set_visible(False)
ax1a.spines['bottom'].set_visible(False)  
ax1a.get_xaxis().set_ticks([])
ax1a.set_ylabel('LFP amplitude (AU)',color = 'k')
ax1a.locator_params(axis='y', nbins=3)
ax1a.plot([200,300],[-2.7,-2.7],linewidth = 3,color = 'k')

ax1a.text(250,-3.0, '500 s',  fontsize=10,horizontalalignment='center',verticalalignment='center')  
tc = plt.title('A', loc = 'left')

ax1b = ax1a.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax1b.set_ylabel('Permittivity variable z (AU)',color = color)  # we already handled the x-label with ax1
ax1b.plot(tvec, xstates[2,:],color = color,linewidth = 2)
ax1b.tick_params(axis='y',labelcolor = color)
ax1b.autoscale(enable=True, axis='x', tight=True)
ax1b.spines['top'].set_visible(False)
ax1b.spines['bottom'].set_visible(False)
ax1b.locator_params(axis='y', nbins=3)


ax1c = plt.subplot(grid1[0,-1],projection = "3d") 
ax1c.plot(trajectory[0,:],trajectory[1,:],trajectory[2,:],linewidth = 0.5,color = 'k')
ax1c.set_xlabel('$x_1$')
ax1c.set_ylabel('$y_1$')    
ax1c.set_zlabel('z')
ax1c.locator_params(axis='y', nbins=2)
ax1c.locator_params(axis='z', nbins=2)
ax1c.locator_params(axis='x', nbins=2)
tc = plt.title('B', loc = 'left')

#%% plot figure 2 and 3 (with and without stimuli)
fig2 = plt.figure()
grid = plt.GridSpec(2, 2)
ax1 = plt.subplot(grid[0,0:])
color = 'tab:blue'
#ax1.set_xlabel('time (s)')
ax1.set_ylabel('LFP amplitude (AU)',color = color)
ax1.plot(tvec[2048:], simulatedLFP[2048:],color = color)
ax1.tick_params(axis='y',labelcolor = color)
ax1.plot([1760, 1760],[-1, 1],'-.',color = 'r')
ax1.plot([1800, 1800],[-1, 1],'-.',color = 'r')
ax1.plot([7683, 7683],[-1, 1],'-.',color = 'r')
ax1.plot([8073, 8073],[-1, 1],'-.',color = 'r')
ax1.plot([100,600],[-1,-1],linewidth = 3,color = 'k')
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.spines['top'].set_visible(False)
#ax2.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax2.spines['left'].set_visible(False)    
ax1.get_xaxis().set_ticks([])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Permittivity variable z (AU)',color = color)  # we already handled the x-label with ax1
ax2.plot(tvec, xstates[2,:],color = color)
ax2.tick_params(axis='y',labelcolor = color)
ax2.autoscale(enable=True, axis='x', tight=True)

ax2.spines['top'].set_visible(False)
#ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
#ax2.spines['left'].set_visible(False)    
ax2.get_xaxis().set_ticks([])


ax3 = plt.subplot(grid[1,0])
#ax3.set_ylabel()  # we already handled the x-label with ax1
ax3.plot(tvec[int(1760*Fs):int(1800*Fs)], simulatedLFP[int(1760*Fs):int(1800*Fs)])
#ax3.tick_params(axis='y',labelcolor = color)    
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)    
ax3.get_xaxis().set_ticks([])
ax3.get_yaxis().set_ticks([]) 

ax4 = plt.subplot(grid[1,1])
#ax3.set_ylabel()  # we already handled the x-label with ax1
ax4.plot(tvec[int(7683*Fs):int(8073*Fs)], simulatedLFP[int(7683*Fs):int(8073*Fs)])
#ax3.tick_params(axis='y',labelcolor = color)    
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)    
ax4.get_xaxis().set_ticks([])
ax4.get_yaxis().set_ticks([])       

#%% plot figure 4 (simulate with stimAmp=4 , b, a = signal.butter(3, 1.5*2/Fs, 'high'))
LAWCN_figFeat(groupedFeatsPromed,tFeatsPromed,0.06,0.063)
#%% plot figure 5 (simulate with stimAmp=0 , b, a = signal.butter(3, 0.05*2/Fs, 'high'))
LAWCN_figFeat(groupedFeatsPromed,tFeatsPromed,-0.012,-0.018)


