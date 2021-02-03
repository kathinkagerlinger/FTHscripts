"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction
Also includes function for Ewald Sphere projection

2020
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
            KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
from scipy.stats import linregress
import fth_reconstruction as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import circle
import h5py
import math
import cupy as cp
import cupyx as cx #.scipy.ndimage.convolve1d


#############################################################
#       PHASE RETRIEVAL FUNCTIONS
#############################################################

def PhaseRtrv(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20,BS=[0,0,0],bsmask=0,real_object=False, average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Fourier_last: Boolean, if True the last constraint applied is the Fourier space Constraint, if false the real space constraint
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
        #Phase=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        #Phase=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        print("using phase given")

    guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    #guess=Phase.copy()
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=(BSmask)
    guess_cp=(guess)
    mask_cp=(mask)
    diffract_cp=(diffract)
    
    Best_guess=np.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=np.zeros(average_img)
    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        update = (1-BSmask_cp) *diffract_cp* np.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp
        
        ###REAL SPACE###
        inv = np.fft.fft2(update)
        if real_object:
            inv=np.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* np.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= np.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * np.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*np.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + np.heaviside(np.real(-inv+alpha*prev),1)*np.heaviside(np.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*np.heaviside(np.real(prev-(beta-3)*inv),0)

                        
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        guess_cp=np.fft.ifft2(inv)
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(np.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=np.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp
                    
        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(np.abs(guess_cp),mask_cp)
            Error_supp_list.append(Error_supp)
        
            print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    guess_cp=np.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* np.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp
        #print("FINAL \n",cp.average(diffract**2), cp.average(cp.abs(guess_cp)**2))
    
    guess=(guess_cp)
    #print("FINAL \n",cp.average(diffract**2), np.average(np.abs(guess)**2))

    #return final image
    return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list

#################

def PhaseRtrv_GPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20,BS=[0,0,0],bsmask=0,real_object=False,average_img=10, Fourier_last=True):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Fourier_last: Boolean, if True the last constraint applied is the Fourier space Constraint, if false the real space constraint
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
        #Phase=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        #Phase=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        print("using phase given")

    guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    #guess=Phase.copy()
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
            
    # start phase retrieval cycle
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        update = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
        
        ###REAL SPACE###
        inv = cp.fft.fft2(update)
        if real_object:
            inv=cp.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* cp.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)

                        
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        guess_cp=cp.fft.ifft2(inv)
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp
                    
        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(cp.abs(guess_cp),mask_cp)
            Error_supp_list.append(Error_supp)
        
            print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    guess_cp=cp.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
        #print("FINAL \n",cp.average(diffract**2), cp.average(cp.abs(guess_cp)**2))
    
    guess=cp.asnumpy(guess_cp)
    #print("FINAL \n",cp.average(diffract**2), np.average(np.abs(guess)**2))

    #return final image
    return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list

from scipy import signal

def PhaseRtrv_with_RL(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, RL_freq=25, RL_it=20, Phase=0,seed=False,
       plot_every=20, BS=[0,0,0], bsmask=0, real_object=False, average_img=10, Fourier_last=True, R_apod=None):
    
    '''
    Iterative phase retrieval function, with GPU acceleration and Richardson Lucy algorithm (http://www.nature.com/articles/ncomms1994)
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MCF
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            Fourier_last: Boolean, if True the last constraint applied is the Fourier space Constraint, if false the real space constraint
            R_apod: Radius apodizing the MCF. Apparently works only if R_apod=None
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support, gamma(MCF)
    
     --------
    author: RB 2020
    '''

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
        
    #apodization mask for RL algorithm
    if type(R_apod)==type(None):
        R_apod=1
    elif type(R_apod)==int:
        yy_apod, xx_apod = circle(diffract.shape[0]//2, diffract.shape[0]//2, R_apod)
        R_apod = np.ones(diffract.shape)/2
        R_apod[yy_apod, xx_apod] = 1
        R_apod=np.fft.fftshift(R_apod)
        R_apod=cp.asarray(R_apod)
    else:
        R_apod=np.fft.fftshift(R_apod)
        R_apod=cp.asarray(R_apod)
        
    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='smoothstep':
        start=Nit//50
        end=Nit-Nit//10
        x=np.array(range(Nit))
        y=(x-start)/(end-start)
        Beta=1-(1-beta_zero)*(6*y**5-15*y**4+10*y**3)
        Beta[:start]=1
        Beta[end:]=0
    elif beta_mode=='sigmoid':
        x=np.array(range(Nit))
        x0=Nit//20
        alpha=1/(Nit*0.15)
        Beta=1-(1-beta_zero)/(1+np.exp(-(x-x0)*alpha)) 
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
   
        
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.random.rand(l,n)*np.pi*2 #-np.pi/2
        #Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        print("using phase given")
        #Phase=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))

    
    
    if gamma is not None:
        
        gamma=np.fft.fftshift(gamma)
        gamma_cp=cp.asarray(gamma)
        gamma_cp/=cp.sum((gamma_cp))
        
        
    #guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    guess= Phase.copy()
  
    #previous result
    prev = None
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    
    Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    

    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        if gamma is None:
            #apply fourier domain constraints (only outside BS)
            update = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
        else:
            #update = (1-BSmask_cp) *diffract_cp/cp.sqrt(FFTConvolve(cp.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp
            update = (1-BSmask_cp) *diffract_cp/cp.sqrt(FFTConvolve(cp.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp
            #update = (1-BSmask_cp) *diffract_cp/cp.sqrt(signal.fftconvolve(cp.abs(guess_cp)**2,gamma_cp, mode='same'))* guess_cp + guess_cp * BSmask_cp
            
        #print("update contains nan ",cp.isnan(cp.abs(update)).any())
        #update[cp.abs(update)==cp.nan]=0
        
        ###REAL SPACE###
        inv = cp.fft.fft2(update)
        if real_object:
            inv=cp.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='RAAR':
            inv = inv + (1-mask_cp)*(beta*prev - 2*beta*inv)
            + (beta*prev -2*beta*inv)* mask_cp* cp.where(-2*inv+prev>0,1,0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
            + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)

                        
        prev=inv.copy()
        
        ### FOURIER SPACE ### 
        new_guess=cp.fft.ifft2(inv)
        
        #print("new_guess contains nan ",cp.isnan(np.abs(new_guess)).any())
        
        
        #RL algorithm
        if (gamma is not None) and s>RL_freq and (s%RL_freq==0):
            
            Idelta=2*np.abs(new_guess)**2-np.abs(guess_cp)**2
            #I_exp=(1-BSmask_cp) *cp.abs(diffract_cp)**2 + cp.abs(FFTConvolve(new_guess,gamma_cp))**2 * BSmask_cp
            I_exp=(1-BSmask_cp) *cp.abs(diffract_cp)**2 + FFTConvolve(cp.abs(new_guess)**2,gamma_cp) * BSmask_cp
            #I_exp=(1-BSmask_cp) *cp.abs(diffract_cp)**2 + signal.fftconvolve(cp.abs(new_guess)**2,gamma_cp, mode="same") * BSmask_cp
                
            gamma_cp = RL( Idelta=Idelta,  Iexp = I_exp , gamma_cp=gamma_cp, RL_it=RL_it, mask_apod=R_apod)
            
        guess_cp = new_guess.copy()
        
        
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        
        #Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr = Error_diffract_cp( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * FFTConvolve(cp.abs(new_guess)**2,gamma_cp))
        #Error_diffr = Error_diffract( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * signal.fftconvolve(cp.abs(new_guess)**2,gamma_cp, mode="same"))
        
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp


        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            
            Error_supp = Error_support(cp.abs(guess_cp),mask_cp)
            Error_supp_list.append(Error_supp)
        
            print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    guess_cp=cp.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        if gamma is None:
            #apply fourier domain constraints (only outside BS)
            guess_cp = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
        else:
            guess_cp = (1-BSmask_cp) *diffract_cp/cp.sqrt(FFTConvolve(cp.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp
            #guess_cp = (1-BSmask_cp) *diffract_cp/cp.sqrt(signal.fftconvolve(cp.abs(guess_cp)**2,gamma_cp, mode="same"))* guess_cp + guess_cp * BSmask_cp

    guess=cp.asnumpy(guess_cp)
    #Error_diffr_list=cp.asnumpy(Error_diffr_list)

    #return final image
    if (gamma is None):
        return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list, gamma
    else:
        gamma = cp.asnumpy(gamma_cp)
        return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list, np.fft.ifftshift(gamma)
    
#########

def RL(Idelta, Iexp, gamma_cp, RL_it, mask_apod=1):
    '''
    Iterative algorithm for Richardson Lucy algorithm (http://www.nature.com/articles/ncomms1994)
    INPUT:  Idelta: difference between Intensity at two successive steps
            Iexp: experimental I
            gamma_cp: MCF
            RL_it: number of steps
            mask_apod: mask in case you want MCF==0 outside of a certain radius
           
    OUTPUT: updated gamma
     --------
    author: RB 2020
    '''
    for l in range(RL_it):
        
        I2=Iexp/(FFTConvolve(Idelta,gamma_cp))
        #print("I2 contains nan ",cp.isnan(I2).any())
        #I2/=cp.nansum(I2)
        gamma_cp = (gamma_cp * (FFTConvolve(Idelta[::-1,::-1], I2)))
        #gamma_cp = (gamma_cp * signal.fftconvolve(Idelta[::-1,::-1], Iexp/(signal.fftconvolve(Idelta,gamma_cp, mode="same")), mode="same") * mask_apod)
        
        gamma_cp/=cp.sum((gamma_cp))
        
    gamma_cp*=mask_apod
    gamma_cp/=cp.nansum(gamma_cp)
    #print("gamma out=",cp.sum(gamma_cp))
        
    return gamma_cp

##########

def FFTConvolve(in1, in2):
    '''
    Function to convolve scattering pattern with its MCF. Or any two matrixes
    INPUT:  in1, in2: two images
    OUTPUT: their convolution
     --------
    author: RB 2020
    '''
    
    in1[in1==cp.nan]=0
    in2[in2==cp.nan]=0
    ret = ((cp.fft.ifft2(cp.fft.fft2(in1) * cp.fft.fft2((in2))))) # o è cp.abs???? per alcuni campioni funxiona in un modo, per altri in un altro

    return ret

#########################################################################
#    Amplitude retrieval
#########################################################################

def AmplRtrv_GPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,
       plot_every=20,ROI=[None,None,None,None],BS=[0,0,0],bsmask=0,average_img=10):
    
    '''
    Iterative phase retrieval function, with GPU acceleration, for clearing the hologram from camera artifacts using "Amplitude rerieval"
    Makes sure that the FTH reconstruction is nonzero only inside the given "mask" support, and that its diffraction pattern stays real and positive.
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zero. Must include all obj.holes and reference holes reconstruction. can be obtained by doing the |FFT(mask)|**2 of the mask used for normal phase retrieval, and applying a threshold
            mode: string defining the algorithm to use (ER, RAAR, HIO)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            plot_every: how often you plot data during the retrieval process
            ROI: region of interest of the obj.hole, useful only for real-time imaging during the phase retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image  
            
    OUTPUT: retrieved scattering pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''
    
    #set titles of plotted images
    
    fig, ax = plt.subplots(1,3)   
    

 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Error_supp_list=[]
    

    BSmask=bsmask
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-500)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
    elif beta_mode=='exp':
        Beta=beta_zero+(1-beta_zero)*(1-np.exp(-(step/7)**3))
    elif beta_mode=='linear_to_beta_zero':
        Beta= 1+(beta_zero-1)/Nit*step
    elif beta_mode=='linear_to_1':
        Beta= beta_zero+(1-beta_zero)/Nit*step
      
    #set initial Phase guess
    if type(Phase)==int:
        Phase=np.random.rand(l,n)*np.pi*2 #-np.pi/2
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
    else:
        Phase0=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Phase)))
        Phase=np.angle(Phase0)

    guess = (1-BSmask)*diffract * np.exp(1j * Phase)+ Phase0*BSmask
  
    #previous result
    prev = None
    
    FTHrec=fth.reconstruct(diffract)
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    #mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    
    
    BSmask_cp=cp.asarray(BSmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    FTHrec_cp=cp.asarray(FTHrec)
    cpROI=cp.asarray(ROI)
    
    Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=cp.zeros(average_img)
    
    print('mode=',mode,'    beta_mode=',beta_mode)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
       
        update = guess_cp
        
        inv = (cp.fft.ifft2((update))) #inv is the FTH reconstruction
        
        inv=cp.fft.ifftshift(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #FTH reconstruction condition
        if mode=='ER':
            inv=FTHrec_cp*mask_cp
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #elif mode=='HIO':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        elif mode=='RAAR':
            inv = FTHrec_cp + (1-mask_cp)*(beta*prev - 2*beta*FTHrec_cp)
            + (beta*prev -2*beta*FTHrec_cp)* mask_cp* cp.where(-2*FTHrec_cp+prev>0,1,0)
            #cp.heaviside(cp.real(-2*inv+prev),0)
        #elif mode=='OSS':
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(-cp.real(inv),0)
        #    #smooth region outside support for smoothing
        #    alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
        #    smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
        #    inv= mask_cp*inv + (1-mask_cp)*smoothed
        #elif mode=='CHIO':
        #    alpha=0.4
        #    inv= (prev-beta*inv) + mask_cp*cp.heaviside(cp.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
        #    + cp.heaviside(cp.real(-inv+alpha*prev),1)*cp.heaviside(cp.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        #elif mode=='HPR':
        #    alpha=0.4
        #    inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        #    + mask_cp*(prev - (beta+1) * inv)*cp.heaviside(cp.real(prev-(beta-3)*inv),0)

                         
        prev=inv.copy()
        
        inv=cp.fft.fftshift(inv)
        
         #apply real and positive diffraction pattern constraint
        guess_cp = np.maximum(cp.real( cp.fft.fft2(inv) ) , cp.zeros(guess_cp.shape))
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess_cp

        
        #COMPUTE ERRORS
        if s % plot_every == 0:
            clear_output(wait=True)
            
            Error_supp = Error_support(cp.abs(guess_cp),mask_cp)
            #Error_supp_list.append(Error_supp)
            
            #ax1.scatter(s,Error_diffr,marker='o',color='red')
            #ax1bis.scatter(s,Error_supp,marker='x',color='blue')
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            guessplot=np.fft.fftshift(cp.asnumpy(guess_cp))
            
            im=(np.fft.ifft2((guessplot)))
            
            im_real=np.real(im)
            mir, mar = np.percentile(im_real[ROI[2]:ROI[3],ROI[0]:ROI[1]], (0,100))
            print(mir,mar)
            im_imag=np.imag(im)
            
            ax[0].imshow((guessplot), cmap='coolwarm')
            #ax3.imshow(im_abs, cmap='binary')

            real_detail=im_real#[cpROI[2]:cpROI[3],cpROI[0]:cpROI[1]]
            imag_detail=im_imag#[cpROI[2]:cpROI[3],cpROI[0]:cpROI[1]]
            ax[1].imshow(real_detail,vmin=mir,vmax=mar)
            ax[2].imshow(imag_detail, cmap='hsv',vmin=-cp.pi,vmax=cp.pi)

            display(plt.gcf())
        
            print(cp.sum(guess_cp),'#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)

    #sum best guess images
    #guess_cp==cp.sum(Best_guess,axis=0)/average_img
    
    guess=cp.asnumpy(guess_cp)

    #return final image
    return np.fft.ifftshift(guess) , Error_diffr_list, Error_supp_list

#############################################################
#    FILTER FOR OSS
#############################################################
def W(npx,npy,alpha=0.1):
    '''
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: number of pixels on the image
            alpha: width of the gaussian 
            
    OUTPUT: gaussian matrix
    
    --------
    author: RB 2020
    '''
    Y,X = np.meshgrid(range(npy),range(npx))
    k=(np.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return np.fft.fftshift(np.exp(-0.5*(k/alpha)**2))

#############################################################
#    ERROR FUNCTIONS
#############################################################
def Error_diffract(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=(diffract-guess)**2
    Den=diffract**2
    Error = Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

def Error_diffract_cp(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=(diffract-guess)**2
    Den=diffract**2
    Error = Num.sum()/Den.sum()#cp.sum(Num)/cp.sum(Den)
    Error=10*cp.log10(Error)
    return Error

def Error_support(prev,mask):
    '''
    Error on the support of retrieved data. 
    INPUT:  prev: retrieved image
            mask: support mask
            
    OUTPUT: Error on the support, how much prev is outside of "mask"
    
    --------
    author: RB 2020
    '''
    Num=prev*(1-mask)**2
    Den=prev**2
    Error=Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

#############################################################
#    function for saving Hdf5 file
#############################################################

"""
functions to create and read hdf5 files.
groups will be converted to dictionaries, containing the data
supports nested dictionaries.

to create hdf file:
    create_hdf5(dict0,filename) where dict0 is the dictionary containing the data and filename the file name
to read hdf file:
    data=cread_hdf5(filename) data will be a dictionary containing all information in "filename.hdf5"
riccardo 2020

"""

def read_hdf5(filename, extension=".hdf5", print_option=True):
    
    f = h5py.File(filename+extension, 'r')
    dict_output = readHDF5(f, print_option = print_option, extension=extension)
    
    return dict_output
    
def readHDF5(f, print_option=True, extension=".hdf5", dict_output={}):
    
    for i in f.keys():
        
    
        if type(f[i]) == h5py._hl.group.Group:
            if print_option==True:
                print("### ",i)
                print("---")
            dict_output[i]=readHDF5(f[i],print_option=print_option,dict_output={})
            if print_option==True:
                print("---")
        
        else:
            dict_output[i]=f[i][()]
            if print_option==True:
                print("•",i, "                  ", type(dict_output[i]))
        
        
    return dict_output
    
def create_hdf5(dict0,filename, extension=".hdf5"):
    
    f=createHDF5(dict0,filename, extension=extension)
    f.close()


def createHDF5(dict0,filename, extension=".hdf5",f=None):
    '''creates HDF5 data structures strating from a dictionary. supports nested dictionaries'''
    print(dict0.keys())
    
#    try:
#        f = h5py.File(filename+ ".hdf5", "w")
#        print("ok")
#    except OSError:
#        print("could not read")
    
    if f==None:
         f = h5py.File(filename+ extension, "w")
    
    
    if type(dict0) == dict:
        
        for i in dict0.keys():
            
            print("create group %s"%i)
            print("---")
            print(i,",",type(dict0[i]))

            if type(dict0[i]) == dict:
                print('dict')
                grp=(f.create_group(i))
                createHDF5(dict0[i],filename,f=grp)
                
            elif type(dict0[i]) == np.ndarray:
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
                
            elif (dict0[i] != None):
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
            print("---")
    return f

#############################################################
#    Ewald sphere projection
#############################################################

from scipy.interpolate import griddata

def inv_gnomonic_2(CCD, center=None, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, method='cubic' , mask=None):
    '''
    Projection on the Ewald sphere for close CCD images. Only gets the new positions on the new projected array and then interpolates them on a regular matrix
    Input:  CCD: far-field diffraction image
            z: camera-sample distance,
            center_y,center_x: pixels in excess we want to add to the borders by zero-padding so that the projected image has existing pixels to use
            px_size: size of CCD pixels
    Output: Output: projected image
    
    -------
    author: RB Nov2020
    '''
    
    # we have to caculate all new angles
    
    #points coordinates positions
    z=experimental_setup['ccd_dist']
    px_size=experimental_setup['px_size']
    if type(center)==type(None):
        center=np.array([CCD.shape[1]/2, CCD.shape[0]/2])
        
    #if type(mask) != type(None):
    #    CCD[mask==1]=np.nan

    print("center=",center, "z=",z )
    values=CCD.flatten()
    points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))).astype('float64')
    
    points[0,:]-=center[0]
    points[1,:]-=center[1]
    points*= px_size
    
    
    #points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))- CCD.shape[0]/2) * px_size

    points=points.T
    
    #now we have to calculate the new points
    points2=np.zeros(points.shape)
    points2[:,0]= z* np.sin( np.arctan( points[:,0] / np.sqrt( points[:,1] **2 + z**2 ) ) )
    points2[:,1]= z* np.sin( np.arctan( points[:,1] / np.sqrt( points[:,0] **2 + z**2 ) ) )

    
    CCD_projected = griddata( points2, values, points, method=method)
    
    CCD_projected = np.reshape(CCD_projected, CCD.shape)
    
    #makes outside from nan to zero
    CCD_projected=np.nan_to_num(CCD_projected, nan=0.0, posinf=0, neginf=0)
    

    return CCD_projected, points2, points, values

############################################
## Fourier Ring Correlation
############################################

def FRC0(im1,im2,width_bin):
    '''
    implements Fourier Ring Correlation. (https://www.nature.com/articles/s41467-019-11024-z)
    Input:  im1,im2: two diffraction patterns with different sources of noise. Can also use same image twice, sampling only odd/even pixels
            width_bin: width of circles we will use to have our histogram
            
    Output: sum_num: array of all numerators value of correlation hystogram
            sum_den: array of all denominators value of correlation hystogram
    
    -------
    author: RB 2020
    '''
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])
    
    for i in range(Num_bins):
        annulus = np.zeros(shape)
        yy_outer, xx_outer = circle(center[1], center[0], (i+1)*width_bin)
        yy_inner, xx_inner = circle(center[1], center[0], i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum( im1* np.conj(im2) * annulus )#np.sum( im1[np.nonzero(annulus)] * np.conj(im2[np.nonzero(annulus)]) )
        sum_den[i]=np.sqrt( np.sum(np.abs(im1)**2* annulus) * np.sum(np.abs(im2)**2* annulus) )
        
    return sum_num,sum_den

def FRC(im1,im2,width_bin):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])
    
    FT1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im1)))
    FT2=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im2)))
    
    for i in range(Num_bins):
        annulus = np.zeros(shape)
        yy_outer, xx_outer = circle(center[1], center[0], (i+1)*width_bin)
        yy_inner, xx_inner = circle(center[1], center[0], i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum( FT1* np.conj(FT2) * annulus )#np.sum( im1[np.nonzero(annulus)] * np.conj(im2[np.nonzero(annulus)]) )
        sum_den[i]=np.sqrt( np.sum(np.abs(FT1)**2* annulus) * np.sum(np.abs(FT2)**2* annulus) )
        
    return sum_num,sum_den

def FRC_1image(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even'''
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    sum_num=np.zeros((4,Num_bins))
    sum_den=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        sum_num[j,:],sum_den[j,:]=FRC(im[FT1st[j]],im[FT2nd[j]],width_bin)

    FRC_array=sum_num/sum_den        
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    
def FRC_GPU(im1,im2,width_bin):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC istogram array
    
    RB 2020'''
    
    im1_cp=cp.asarray(im1)
    im2_cp=cp.asarray(im2)
    
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=cp.zeros(Num_bins)
    sum_den=cp.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])
    
    FT1=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im1)))
    FT2=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im2)))
    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = circle(center[1], center[0], (i+1)*width_bin)
        yy_inner, xx_inner = circle(center[1], center[0], i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=cp.sum( FT1* cp.conj(FT2) * annulus )
        sum_den[i]=cp.sqrt( cp.sum(cp.abs(FT1)**2* annulus) * cp.sum(cp.abs(FT2)**2* annulus) )
        
    FRC_array=sum_num/sum_den
    FRC_array_np=cp.asnumpy(FRC_array)
    
    return FRC_array_np


def FRC_1image_GPU(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even
    
    RB 2020'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    FRC_array=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        FRC_array[j,:]=FRC_GPU(im[FT1st[j]],im[FT2nd[j]],width_bin)
      
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    
def half_bit_thrs(im, SNR=0.5, width_bin=5):
    '''van heel and schatz 2005
    gives you an array containing values for the half bit threshold
    RB 2020'''
    
    shape=im.shape
    Num_bins=shape[0]//(2*width_bin)
    center = np.array([shape[0]//2, shape[1]//2])
    thr=np.zeros(Num_bins)
    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = circle(center[1], center[0], (i+1)*width_bin)
        yy_inner, xx_inner = circle(center[1], center[0], i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0
        n=np.sum(annulus) #counting...
        #print(n)
    
        thr[i]=(SNR+ (2*np.sqrt(SNR)+1)/np.sqrt(n))/(SNR+1+2*np.sqrt(SNR)/np.sqrt(n))
    return thr

def PRTF(im, exp_im, width_bin=5):
    '''function for Phase Retrieval Transfer Function
    RB Jan 2021
    INPUT: im: sums of retrieved image
            exp_im: experimental scattering pattern
            width_bin: width of bins used to plot PRTF
    output: prtf: phase retrieval transfer function'''
    
    prtf= im/exp_im
    
    prtf_cp=cp.asarray(prtf)
    
    shape=prtf.shape
    Num_bins=shape[0]//(2*width_bin)
    
    prtf_array=cp.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])

    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = circle(center[1], center[0], (i+1)*width_bin)
        yy_inner, xx_inner = circle(center[1], center[0], i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        prtf_array[i]=cp.sum( prtf_cp * annulus )/cp.sum(annulus)
        
    prtf_array_np=cp.asnumpy(prtf_array)
    
    return prtf_array_np
    

#############################################################
#       AUTO SHIFTING
#############################################################
def realign(FT1, FT2, max_shift=20, plot=True):
    '''automaticly realigns images with sub-pixel shiftint 
    RB November 2020
    
    INPUT:  FT1: image to realign in F space
            FT2: ref image
            crop: defines the portion of fourier space you consider
            
    output: FT1_shifted:  FT1 shifted as FT2
            dx: # pixels shifted x axis
            dy: # pixels shifted y axis
    
    RB 2020'''
    
    
    npx,npy=FT1.shape
    
    crop=npx//2 - npx//(2*max_shift)-1
    
    #FT1=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im1)))
    #FT2=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im2)))
    
    angle1=np.angle(FT1)[crop:-crop,crop:-crop]
    angle2=np.angle(FT2)[crop:-crop,crop:-crop]
    
    phase_profile_x=np.mean(np.exp(1j*(angle1-angle2)),axis=1)
    phase_profile_y=np.mean(np.exp(1j*(angle1-angle2)),axis=0)
    
    x=np.linspace(0,len(phase_profile_y)-1,len(phase_profile_y))
    
    slope_x, intercept_x,_,_,_=linregress(x, np.angle(phase_profile_x))
    slope_y, intercept_y,_,_,_=linregress(x, np.angle(phase_profile_y))
    
    Ky,Kx = np.meshgrid(range(npx),range(npx))
    
    dx=slope_x*npx/(2*np.pi)
    dy=slope_y*npy/(2*np.pi)
    
    FT1_shifted=FT1*np.exp(-1j*(slope_y*Ky+slope_x*Kx))*np.exp(-1j*np.average(angle1-angle2))
    #+(intercept_x+intercept_y)
    
    print("image shifted of dx=%f, dy=%f"%(dx,dy))
    
    #im1_shifted=fth.reconstruct(FT1_shifted)
    
    if plot==True:
        fig,ax=plt.subplots()
        ax.plot(np.angle(phase_profile_y),'o',label='phase_y')
        ax.plot(np.angle(phase_profile_x),'o',label='phase_x')
        ax.plot(slope_x*x+intercept_x,label='slope_x')
        ax.plot(slope_y*x+intercept_y,label='slope_x')
        plt.legend()
    
    return FT1_shifted, dx, dy

from scipy import signal

def correlate_realign2(FT, x):
    '''cross-correlation'''
    
    npy,npx=FT.shape
    v,u = np.meshgrid(range(npx),range(npx))
    v-=npx//2
    u-=npy//2
    #corr = signal.correlate(FT1/FT1.std(), FT2/FT2.std(), mode='same') / FT1.size
    
    corr=np.sum( FT * np.exp(1j*2*np.pi*(u*x[0]/npx + v*x[1]/npy)) )
    
    return corr

def derivative_realign2(FT, x):
    '''derivative of cross-correlation'''
    FT=np.conjugate(FT)
    npy,npx=FT.shape
    v,u = np.meshgrid(range(npx),range(npx))
    v-=npx//2
    u-=npy//2
    axis=0
    der=np.zeros((2))
    corr=(correlate_realign2(FT,x))
    exp=2*np.pi/npx* np.exp(-1j*2*np.pi*(u*x[0]/npx+ v*x[1]/npy) )
    
    der[0]=2*np.imag(corr * np.sum( u * FT * exp ))
    der[1]=2*np.imag(corr * np.sum( v * FT * exp ))
    
    return der
    
def find_max_realign2(FT, x=np.array([0,0]), imax = 10, eps = 0.01):
    '''algorithm to realign one image following a reference.
    images must have been prentively aligned using upsampled FFT with usmpling factor k=2
    (from "Efficient subpixel image registration algorithms" 10.1364/OL.33.000156)
    RB 2021'''
    
    #corr=np.sum(correlate_realign2(FT1,FT2))
    #corr = signal.correlate(FT1/FT1.std(), FT2/FT2.std(), mode='same') / FT1.size

    x = x+ np.array(FT1.shape)//2

    i = 0

    r = derivative_realign2(FT1, FT2, x)
    print("r=",r)
    d = r
    deltanew = np.dot(r.T ,r)
    print("deltanew=",deltanew)
    delta0 = deltanew
    
    while i < imax and deltanew > eps**2 * delta0:
        
        alpha = float(deltanew / float(np.dot(d.T , np.dot(np.sum(correlate_realign2(FT1,FT2,x)), d))))
        x = x + alpha * d
        
        #steps.append((x[0, 0], x[1, 0]))
        
        r = derivative_realign2(FT1, FT2, x)
        deltaold = deltanew
        deltanew = np.dot(r.T ,r)
        
        beta = float(deltanew / float(deltaold))
        d = r + beta * d
        i += 1
        
        print("i=",i,", d=",d,", x=",x)
    return x

from scipy import optimize
def find_max_realign3(FT, x=np.array([0,0]), imax = 10, eps = 0.01, bnds=None):
    '''algorithm to realign one image following a reference.
    images must have been prentively aligned using upsampled FFT with usmpling factor k=2
    (from "Efficient subpixel image registration algorithms" 10.1364/OL.33.000156)
    RB 2021'''
    
    def jacobian(x):
        return -derivative_realign2(FT, x)
    def f(x):
        return -np.abs((correlate_realign2(FT, x)))**2
    
    if bnds == None:
        bnds = ((x[0]-2, x[0]+2), (x[1]-2, x[1]+2))

    res=optimize.minimize(f, x0=x, method="cg", bounds=bnds, jac=jacobian)
    print(res)
    x=res.x
    return x


def realign2(FT1,FT2, x0=np.array([0,0]), bnds=None):
    '''algorithm to realign one image following a reference.
    images must have been prentively aligned using upsampled FFT with usmpling factor k=2
    (from "Efficient subpixel image registration algorithms" 10.1364/OL.33.000156)
    RB 2021'''
    
    #let's first align with 1 pixel refinement
    #compute cross-correlation and find its maximum
    image1=fth.reconstructCDI(FT1)
    image2=fth.reconstructCDI(FT2)
    corr = signal.correlate(image1/image1.std(), image2/image2.std(), mode='same')/(image1.size)
    
    print(np.amax(np.abs(corr)))
    x_shift=np.argmax(corr)%corr.shape[0]- corr.shape[1]//2
    y_shift=np.argmax(corr)//corr.shape[0]- corr.shape[0]//2
    print("first shift of ( %d , %d )"%(-x_shift,-y_shift))
    FT1=fth.sub_pixel_centering(FT1, dx=-x_shift, dy=-y_shift)
    
    FT=FT1*np.conjugate(FT2)
    
    if bnds==None:
        bnds = ((-1, +1), (-1, +1))

    x= find_max_realign3(FT, x=x0, imax = 10, eps = 0.01, bnds=bnds)
    print("x=", x)

    y_shift=x[0]#-FT.shape[0]
    x_shift=x[1]#-FT.shape[1]
    print("shifting of dx=%f, dy=%f"%(x_shift,y_shift))
    #shift pixels et voila
    FT1=fth.sub_pixel_centering(FT1, dx=x_shift, dy=y_shift)
    
    return FT1, x_shift, y_shift
