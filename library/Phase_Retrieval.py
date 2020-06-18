import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
import fth_reconstruction as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import circle
import h5py
import math


#############################################################
#       PHASE RETRIEVAL FUNCTIONS
#############################################################

def PhaseRtrv(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,
       plot_every=20,ROI=[None,None,None,None],BS=[0,0,0],real_object=False,average_img=10):#,bs_mask=np.zeros((diffract.shape[0],diffract.shape[1]))):
    '''Iterative phase retrieval function
        diffract=far field data, mask=support , mode=algorithm type, Nit= number of step
        beta_zero,beta_mode = evolution of beta parameter, Phase=initial image to start from,
        ROI=region of interest (Obj.Hole), plotted during retrieval, BS=(centery,centerx,radius) of BeamStopper,
        real_object=possibility of only real image
        plot_every= how often you plot data
        average_img=number of image to be averaged
        
        returns in output retrieved image
    '''

    #set titles of plotted images
    fig=plt.figure()
    gs=GridSpec(1,2, left=0, right=0.25) # 2 rows, 3 columns
    #fig, ax = plt.subplots(3,2)   
    ax1=fig.add_subplot(gs[:,:]) # First column, all rows
    title0='Error plot (diffr_sim-diffr_exp)'
    color = 'tab:red'
    ax1.set_xlabel('step')
    ax1.set_ylabel('Err diffr', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1bis = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax1bis.set_ylabel('Err supp', color=color)  # we already handled the x-label with ax1
    ax1bis.tick_params(axis='y', labelcolor=color)
    
    gs2=GridSpec(2,2,left=0.45, right=1) # 2 rows, 3 columns
    ax2=fig.add_subplot(gs2[0,0]) # First row, second column
    ax3=fig.add_subplot(gs2[1,0]) # second row, second column
    ax4=fig.add_subplot(gs2[0,1]) # First row, third column
    ax5=fig.add_subplot(gs2[1,1]) # second row, third column
 
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    Error_diffr_list=[]
    Best_guess=np.zeros((average_img,l,n),dtype = 'complex_')
    Best_error=np.zeros(average_img)
    BSmask=np.zeros((l,n))
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    #BSmask+=bs_mask
    
    #prepare beta function
    step=np.arange(Nit)
    if beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-900)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
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
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        if BS_radius==0:
            update = diffract * np.exp(1j * np.angle(guess))
        else:
            update = (1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask
        
        inv = np.fft.fft2(update)
        if real_object:
            inv=np.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask
        elif mode=='HIOs':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
        elif mode=='RAAR':
            inv = inv + (1-mask)*(beta*prev - 2*beta*inv)
            + mask*(beta*prev -2*beta*inv)* np.heaviside(np.real(-2*inv+prev),0)
        elif mode=='OSS':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= np.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * np.fft.fft2(inv))          
            inv= mask*inv + (1-mask)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask*np.heaviside(np.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + np.heaviside(np.real(-inv+alpha*prev),1)*np.heaviside(np.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask)*(prev - (beta+1) * inv)
            + mask*(prev - (beta+1) * inv)*np.heaviside(np.real(prev-(beta-3)*inv),0)

                         
        prev=inv.copy()
        guess = np.fft.ifft2(inv)
        
        #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
        Error_diffr = Error_diffract(np.abs(guess)*(1-BSmask),diffract*(1-BSmask))
        Error_diffr_list.append(Error_diffr)
        if s>2:
            if Error_diffr<=Error_diffr_list[s-1] and Error_diffr<=Error_diffr_list[s-2]:
                j=np.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j]=Error_diffr
                    Best_guess[j,:,:]=guess

        
        #save an image of the progress
        if s % plot_every == 0:
            clear_output(wait=True)
            
            #compute error
            Error_supp = Error_support(np.abs(guess),mask)
            
            ax1.scatter(s,Error_diffr,marker='o',color='red')
            ax1bis.scatter(s,Error_supp,marker='x',color='blue')
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            
            im=np.fft.ifftshift(inv)
            
            im_abs=np.abs(im)
            im_angle=np.angle(im)
            ax2.imshow(np.log10(np.abs(np.fft.ifftshift(guess))), cmap='coolwarm')
            ax3.imshow(im_abs, cmap='binary')

            abs_detail=im_abs[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            angle_detail=im_angle[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            ax4.imshow(abs_detail, cmap='binary')
            ax5.imshow(angle_detail, cmap='hsv',vmin=-np.pi,vmax=np.pi)

            display(plt.gcf())
        
            print('step:',s,'   beta=',beta,'   alpha=',alpha, '    mode=',mode,
                  '    beta_mode=',beta_mode,'   N_avg=',average_img)

    #sum best guess images
    guess==np.sum(Best_guess,axis=0)/average_img

    #return final image
    if BS_radius==0:
        return np.fft.ifftshift(np.fft.fft2(diffract * np.exp(1j * np.angle(guess)))), diffract * np.exp(1j * np.angle(guess))
    else:
        return np.fft.ifftshift(np.fft.fft2((1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask)), (1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask
    

def PhaseRtrv_fast(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,
       plot_every=20,ROI=[None,None,None,None],BS=[0,0,0],real_object=False):
    '''Iterative phase retrieval function
        diffract=far field data, mask=support , mode=algorithm type, Nit= number of step
        beta_zero,beta_mode = evolution of beta parameter, Phase=initial image to start from, random Phase is Phase=0,
        ROI=region of interest (Obj.Hole), plotted during retrieval, BS=(center_y,center_x,radius) of BeamStopper,
        real_object=possibility of only real image
        plot_every= how often you plot data
        
        returns in output retrieved image

        Riccardo Battistelli, 01/06/20
    '''

    #set titles of plotted images
    fig=plt.figure()
    gs=GridSpec(1,2, left=0, right=0.25) # 2 rows, 3 columns
    #fig, ax = plt.subplots(3,2)   
    ax1=fig.add_subplot(gs[:,:]) # First column, all rows
    title0='Error plot (diffr_sim-diffr_exp)'
    color = 'tab:red'
    ax1.set_xlabel('step')
    ax1.set_ylabel('Err diffr', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1bis = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax1bis.set_ylabel('Err supp', color=color)  # we already handled the x-label with ax1
    ax1bis.tick_params(axis='y', labelcolor=color)
    
    gs2=GridSpec(2,2,left=0.45, right=1) # 2 rows, 3 columns
    ax2=fig.add_subplot(gs2[0,0]) # First row, second column
    ax3=fig.add_subplot(gs2[1,0]) # second row, second column
    ax4=fig.add_subplot(gs2[0,1]) # First row, third column
    ax5=fig.add_subplot(gs2[1,1]) # second row, third column

    
    
    #set parameters and BSmask
    (l,n) = diffract.shape
    alpha=None
    BSmask=np.zeros((l,n))
    BS_radius=BS[2]
    yy, xx = circle(BS[1], BS[0], BS_radius)
    if BS_radius>0:
        BSmask[yy, xx] = 1
    
    #prepare beta function
    step=np.arange(Nit)
    if beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-0.45*Nit)/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
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
    
    #shift everything to the corner
    BSmask=np.fft.fftshift(BSmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    for s in range(0,Nit):

        beta=Beta[s]
        
        #apply fourier domain constraints (only outside BS)
        if BS_radius==0:
            update = diffract * np.exp(1j * np.angle(guess))
        else:
            update = (1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask
        
        inv = np.fft.fft2(update)
        if real_object:
            inv=np.real(inv)
        
        if prev is None:
            prev=inv.copy()
            
        #support Projection
        if mode=='ER':
            inv=inv*mask
        elif mode=='HIOs':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
        elif mode=='RAAR':
            inv = inv + (1-mask)*(beta*prev - 2*beta*inv)
            + mask*(beta*prev -2*beta*inv)* np.heaviside(np.real(-2*inv+prev),0)
        elif mode=='OSS':
            inv =  inv + (1-mask)*(prev - (beta+1) * inv) + mask*(prev - (beta+1) * inv)*np.heaviside(-np.real(inv),0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= np.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * np.fft.fft2(inv))          
            inv= mask*inv + (1-mask)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask*np.heaviside(np.real(inv-alpha*prev),1)*(-prev+(beta+1)*inv)
            + np.heaviside(np.real(-inv+alpha*prev),1)*np.heaviside(np.real(inv),1)* ((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            alpha=0.4
            inv =  inv + (1-mask)*(prev - (beta+1) * inv)
            + mask*(prev - (beta+1) * inv)*np.heaviside(np.real(prev-(beta-3)*inv),0)

                         
        prev=inv.copy()
        guess = np.fft.ifft2(inv)

        
        #save an image of the progress
        if s % plot_every == 0:
            clear_output(wait=True)
            Error_diffr = Error_diffract(np.abs(guess)*(1-BSmask),diffract*(1-BSmask))
            Error_supp = Error_support(np.abs(guess),mask)
            
            ax1.scatter(s,Error_diffr,marker='o',color='red')
            ax1bis.scatter(s,Error_supp,marker='x',color='blue')
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            
            im=np.fft.ifftshift(inv)
            
            im_abs=np.abs(im)
            im_angle=np.angle(im)
            ax2.imshow(np.abs(np.fft.ifftshift(guess)), cmap='RdBu')
            
            ax3.imshow(im_abs, cmap='binary')

            abs_detail=im_abs[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            angle_detail=im_angle[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            ax4.imshow(abs_detail, cmap='binary')
            ax5.imshow(angle_detail, cmap='RdBu',vmin=-np.pi/2,vmax=np.pi/2)
            

            display(plt.gcf())
        
            print('step:',s,'   beta=',beta,'   alpha=',alpha, '    mode=',mode)

    #apply fourier domain constraints (only outside BS)
    if BS_radius==0:
        return np.fft.ifftshift(np.fft.fft2(diffract * np.exp(1j * np.angle(guess))))
    else:
        return np.fft.ifftshift(np.fft.fft2((1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask))

#############################################################
#    FILTER FOR OSS
#############################################################
def W(npx,npy,alpha=0.1):
    Y,X = np.meshgrid(range(npy),range(npx))
    k=(np.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return np.fft.fftshift(np.exp(-0.5*(k/alpha)**2))

#############################################################
#    ERROR FUNCTIONS
#############################################################
def Error_diffract(guess, diffract):
    Num=(diffract-guess)**2
    Den=diffract**2
    Error = Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

def Error_support(prev,mask):
    Num=prev*(1-mask)**2
    Den=prev**2
    Error=Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

#############################################################
#    function for setting PR parameters using widgets
#############################################################
def widgParam(BS):
    def f(N_step,algorithm,beta,beta_func,only_real,BeamStop,N_average_images,plot_how_often):
        global BS,plot_every,mode,beta_mode,real_object,average,Nit,beta_zero
        if BeamStop==False:
            BS=[0,0,0]
        plot_every=plot_how_often
        mode = algorithm
        beta_mode=beta_func
        real_object = only_real
        average=N_average_images
        Nit=N_step
        beta_zero=beta
        return (Nit,mode,beta_mode,real_object,plot_every,BS,average,beta_zero)
        
    widgets.interact(f, N_step=widgets.IntSlider(min=0, max=3000, step=5, value=200),
        algorithm=['ER','RAAR','HIO','CHIO','HIOs','OSS','HPR'],
         beta=widgets.FloatSlider(value=0.8,min=0.5,max=1.0,step=0.01),
         beta_func=['const','arctan','exp','linear_to_beta_zero','linear_to_1'],
         only_real=False, BeamStop=False,
         N_average_images=widgets.IntSlider(min=1, max=30, step=1, value=10),
         plot_how_often=widgets.IntSlider(min=5, max=300, step=5, value=50));
    
    return (Nit,mode,beta_mode,real_object,plot_every,BS,average,beta_zero)

#############################################################
#    save parameters
#############################################################

def save_retrieval(fname, nr_nobs, nr_sbs, nr_lbs, retrieved_holo_p, retrieved_holo_n, prop_dist, phase, roi, comment = ''):
    '''
    Save everything in a hdf file. If the file already exists, append the reconstruction and parameters to that file (key is always reco%increasing number)
    INPUT:  fname: path and name of the hdf file
            nr_nobs, nr_sbs, nr_lbs: entry numbers for the data without, with small, with large beamstop
            retrieved_holo_p, retrieved_holo_n: retrieved holograms for positive, negative helicity
            recon: beamstop diameter
            prop_dist: propagation distance
            phase: phase
            roi: ROI coordinates
            comment: string if you want to leave a comment about this reconstruction, default is an empty string
    -------
    author: RB 2020
    '''
    
    reco_dict = {
        'entry numbers no beamstop': nr_nobs,
        'entry numbers small beamstop': nr_sbs,
        'entry numbers large beamstop': nr_lbs,
        'retrieved hologram positive helicity': retrieved_holo_p,
        'retrieeved hologram negative helicity': retrieved_holo_n,
        'ROI coordinates': roi,
        'Propagation distance': prop_dist,
        'phase': phase,
        'comment': comment
    }
    
    save_reco_dict_to_hdf(fname, reco_dict)
    return


def save_stitching(fname, nr_nobs, nr_sbs, nr_lbs, c_nobs, c_bs, r_bs1, r_bs2, fac1, sig1, shift1, fac2, sig2, shift2, stitched_holo_p, stitched_holo_n, recon, prop_dist, phase, roi, comment = ''):
    '''
    Save everything in a hdf file. If the file already exists, append the reconstruction and parameters to that file (key is always reco%increasing number)
    INPUT:  fname: path and name of the hdf file
            nr_nobs, nr_sbs, nr_lbs: entry numbers for the data without, with small, with large beamstop
            c_nobs, c_bs: center for the images without, with beamstop
            r_bs1, r_bs2: radius of small, large beamstop
            fac1, sig1, shift1; fac2, sig2, shift2: factor, sigma of Gauss and shift for the first and second stitching
            stitched_holo_p, stitched_holo_n: stitched holograms for positive, negative helicity
            recon: beamstop diameter
            prop_dist: propagation distance
            phase: phase
            roi: ROI coordinates
            comment: string if you want to leave a comment about this reconstruction, default is an empty string
    -------
    author: KG 2020
    '''
    
    reco_dict = {
        'entry numbers no beamstop': nr_nobs,
        'entry numbers small beamstop': nr_sbs,
        'entry numbers large beamstop': nr_lbs,
        'center no beamstop': c_nobs,
        'center with beamstop': c_bs,
        'radius small beamstop': r_bs1,
        'factor first stitching': fac1,
        'sigma first stitching': sig1,
        'shift first stitching': shift1,
        'radius large beamstop': r_bs2,
        'factor second stitching': fac2,
        'sigma second stitching': sig2,
        'shift second stitching': shift2,
        'stitched hologram positive helicity': stitched_holo_p,
        'stitched hologram negative helicity': stitched_holo_n,
        'ROI coordinates': roi,
        'Propagation distance': prop_dist,
        'phase': phase,
        'comment': comment
    }
    
    save_reco_dict_to_hdf(fname, reco_dict)
    return


def save_reco_dict_to_hdf(fname, reco_dict):
    '''Saves a flat dictionary to a new hdf group in given file.
    
    Parameters
    ----------
    fname : str
        hdf file name
    reco_dict : dict
        Flat dictionary
    
    Returns
    -------
    grp : str
        Name of the new data group.
    -------
    author: dscran 2020
    '''
    with h5py.File(fname, mode='a') as f:
        i = 0
        while f'reco{i:02d}' in f:
            i += 1
        for k, v in reco_dict.items():
            f[f'reco{i:02d}/{k}'] = v
    return f'reco{i:02d}'

def read_hdf(fname):
    '''
    reads the latest saved parameters in the hdf file
    INPUT:  fname: path and filename of the hdf file
    OUtPUT: image numbers, retrieved_p, retrieved_n, recon, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    i = 0
    while f'reco{i:02d}' in f:
        i += 1
    i -= 1
    
    nobs_numbers = f[f'reco{i:02d}/entry numbers no beamstop'].value
    sbs_numbers = f[f'reco{i:02d}/entry numbers small beamstop'].value
    lbs_numbers = f[f'reco{i:02d}/entry numbers large beamstop'].value
    retrieved_holo_p = f[f'reco{i:02d}/retrieved hologram positive helicity'].value 
    retrieved_holo_n = f[f'reco{i:02d}/retrieeved hologram negative helicity'].value 
    prop_dist = f[f'reco{i:02d}/Propagation distance'].value
    phase = f[f'reco{i:02d}/phase'].value
    roi = f[f'reco{i:02d}/ROI coordinates'].value

    return (nobs_numbers, sbs_numbers, lbs_numbers, retrieved_holo_p, retrieved_holo_n, prop_dist, phase, roi)

def read_stitching(fname):
    '''
    reads the latest saved parameters in the hdf file
    INPUT:  fname: path and filename of the hdf file
    OUtPUT: image numbers, retrieved_p, retrieved_n, recon, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: RB 2020
    '''
    f = h5py.File(fname, 'r')
    i = 0
    while f'reco{i:02d}' in f:
        i += 1
    i -= 1
    
    nobs_numbers = f[f'reco{i:02d}/entry numbers no beamstop'].value
    sbs_numbers = f[f'reco{i:02d}/entry numbers small beamstop'].value
    lbs_numbers = f[f'reco{i:02d}/entry numbers large beamstop'].value
    c_nobs = f[f'reco{i:02d}/center no beamstop'].value
    c_bs = f[f'reco{i:02d}/center with beamstop'].value
    r_bs1 = f[f'reco{i:02d}/radius small beamstop'].value
    fac1 = f[f'reco{i:02d}/factor first stitching'].value  
    sig1 = f[f'reco{i:02d}/sigma first stitching'].value
    shift1 = f[f'reco{i:02d}/shift first stitching'].value
    r_bs2 = f[f'reco{i:02d}/radius large beamstop'].value
    fac2 = f[f'reco{i:02d}/factor second stitching'].value
    sig2 = f[f'reco{i:02d}/sigma second stitching'].value
    shift2 = f[f'reco{i:02d}/shift second stitching'].value
    stitched_holo_p = f[f'reco{i:02d}/stitched hologram positive helicity'].value
    stitched_holo_n = f[f'reco{i:02d}/stitched hologram negative helicity'].value
    roi = f[f'reco{i:02d}/ROI coordinates'].value
    prop_dist = f[f'reco{i:02d}/Propagation distance'].value
    phase = f[f'reco{i:02d}/phase'].value


    return (nobs_numbers,sbs_numbers,lbs_numbers,c_nobs,c_bs,r_bs1,fac1,sig1,shift1,r_bs2,fac2,sig2,shift2, stitched_holo_p, stitched_holo_n, roi, prop_dist, phase)

#############################################################
#    Focusing with propagation
#############################################################

def propagate(im_p, im_n, ROI,mask=1, phase=0, prop_dist=0, scale=(0,100), ccd_dist=18e-2, energy=779.5, px_size=20e-6):
    '''
    starts the quest for the right propagation distance and global phase shift.
    reconstructs separate helicities. Plots their ratio.
    Input:  the phase retrieved images as returned by PhaseRtrv
            coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
    Returns the two sliders. When you are finished, you can save the positions of the sliders.
    -------
    author: RB 2020
    '''
    mask=mask[ROI[2]:ROI[3], ROI[0]:ROI[1]]
    holo_p=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im_p)))
    holo_n=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im_n)))

    ph_flip = False
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,2)
    def p(x,y):
        image_p = fth.reconstruct(fth.propagate(holo_p, x*1e-6, ccd_dist = ccd_dist, energy = energy, px_size = px_size)*np.exp(1j*y))
        image_n = fth.reconstruct(fth.propagate(holo_n, x*1e-6, ccd_dist = ccd_dist, energy = energy, px_size = px_size)*np.exp(1j*y))
        
        image=image_p/image_n
        mir, mar = np.percentile(np.real(mask*image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)
        mii, mai = np.percentile(np.imag(mask*image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)

        ax1 = axs[0].imshow(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mir, vmax = mar)
        #fig.colorbar(ax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), cmap='gray', vmin = mii, vmax = mai)
        #fig.colorbar(ax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return
    
    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_prop = widgets.FloatSlider(min=-10, max=10, step=0.01, value=prop_dist, layout=layout, description='propagation[um]', style=style)
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout, description='phase shift', style=style)
    
    widgets.interact(p, x=slider_prop, y=slider_phase)

    #input("Press the <ENTER> key to continue...")
    button = widgets.Button(description="Finished")
    display(button) 
    def on_button_clicked(b):
        slider_prop.close()
        slider_phase.close()
        return
    button.on_click(on_button_clicked)
    
    #ph_flip_button = widgets.Button(description="Flip Phase")
    #display(ph_flip_button)
    #def flip_phase(b):
    #    ph_flip = not(ph_flip)
    #    return
    #button.on_click(flip_phase)
    
    return (slider_prop, slider_phase, button)#, ph_flip_button, ph_flip)

#############################################################
#    Focusing with propagation
#############################################################

def inv_gnomonic(CCD, z=20e-2, center_y=10, center_x=10, px_size=20e-6):
    '''
    Projection on the Ewald sphere for close CCD images.
    Input: far-field diffraction image, z: distance,
    center_y, center_x are added to the side, so the image has to be cropped later
    -------
    author: RB 2020
    '''

    CCD2=np.pad(CCD,((center_x, center_x), (center_y, center_y)), 'constant', constant_values=0)
    height_CCD= px_size*CCD.shape[0]
    width_CCD= px_size*CCD.shape[1]

    A = (height_CCD,width_CCD)

    pix_size = max(A) / max(CCD.shape)       # Pixelsize in meters   
    z = z / pix_size                         # get rid of dimensions ...calculate in pixels
    alpha = np.sin(np.arctan(1/z))            # angle of a single pixel in the projected image prima senza np.sin
    print('alpha_Utils=',alpha)
    #alpha = np.sin(np.arctan(CCD.shape[1]/(2*z))) /CCD.shape[1]*2
    #alpha=np.arctan(max(CCD.shape)/z)/CCD.shape[0]
    lim_y=np.sin(np.arctan(CCD.shape[1]/(2*z))) #prima senza np.sin
    lim_x=np.sin(np.arctan(CCD.shape[0]/(2*z)))
    y_p=0

    CCD=CCD2
    center_y = round(CCD.shape[0]/2)
    center_x = round(CCD.shape[1]/2)
    Output = np.zeros(CCD.shape)             #Output matrix

    for k in range(center_y):
        for l in range(center_x):
          
          y= k*alpha #per modificarlo dovremmo considerare k*alpha=sen theta
          x= l*alpha
          if  np.abs(x) <= lim_x  and np.abs(y/np.sqrt(1-x**2)) <= lim_y:

            #      fourth quadrant pixel positions 
            x_p=np.tan(np.arcsin(x))*np.sqrt(y_p**2+z**2) #solo np.tan(phi), sotto theta             # upper left 
            y_p=np.tan(np.arcsin(y/np.sqrt(1-x**2)))*np.sqrt(x_p**2+z**2)

            x2_p=np.tan(np.arcsin(x+alpha))*np.sqrt(y_p**2+z**2)
            y2_p=np.tan(np.arcsin(y/np.sqrt(1-(x+alpha)**2)))*np.sqrt(x2_p**2+z**2)          # upper right  
        
            y3_p=np.tan(np.arcsin((y+alpha)/np.sqrt(1-(x+alpha)**2)))*np.sqrt(x2_p**2+z**2)
            x3_p=np.tan(np.arcsin(x+alpha))*np.sqrt(y3_p**2+z**2)      # lower right 
        
            x4_p=np.tan(np.arcsin(x))*np.sqrt(y3_p**2+z**2)            # lower left 
            y4_p=np.tan(np.arcsin((y+alpha)/np.sqrt(1-x**2)))*np.sqrt(x4_p**2+z**2)
            
            y_av=round((y_p+y2_p+y3_p+y4_p)/4)
            x_av=round((x_p+x2_p+x3_p+x4_p)/4)
        
            if (y_av > 0 and y_av < center_y and x_av > 0 and x_av < center_x):
              #if l==0:
               # print('doing it for (k.l)= (',k,l,')=',CCD[math.floor(y_p)+center_y,math.floor(x_p)+center_x])
              #       quadrant V 
              a1=abs(x_p-x_av)
              b1=abs(y_p-y_av) # area roughly covered in upper left pixel
              Output[center_y+k,center_x+l]=a1*b1*CCD[math.floor(y_p)+center_y-1,math.floor(x_p)+center_x-1]
        
              a2=abs(x2_p-x_av)
              b2=abs(y2_p-y_av) # ...upper right pixel
              Output[center_y+k,center_x+l]=Output[center_y+k,center_x+l]+a2*b2*CCD[math.floor(y_p)+center_y-1,math.ceil(x_p)+center_x-1]       

              a3=abs(x3_p-x_av)
              b3=abs(y3_p-y_av) # ...lower right pixel
              Output[center_y+k,center_x+l]=Output[center_y+k,center_x+l]+a3*b3*CCD[math.ceil(y_p)+center_y-1,math.ceil(x_p)+center_x-1]      
        
              a4=abs(x4_p-x_av)
              b4=abs(y4_p-y_av) # ...lower left pixel
              Output[center_y+k,center_x+l]=Output[center_y+k,center_x+l]+a4*b4*CCD[math.ceil(y_p)+center_y-1,math.floor(x_p)+center_x-1]   
        
              #       quadrant III 
              Output[center_y+k,center_x-l]=a1*b1*CCD[center_y+math.floor(y_p)-1,center_x-math.floor(x_p)-1]                                     #...upper right pixel
              Output[center_y+k,center_x-l]+=a2*b2*CCD[center_y+math.floor(y_p)-1,center_x-math.ceil(x_p)-1]        #...upper left pixel  
              Output[center_y+k,center_x-l]+=a3*b3*CCD[center_y+math.ceil(y_p)-1,center_x-math.ceil(x_p)-1]         #...lower left pixel  
              Output[center_y+k,center_x-l]+=a4*b4*CCD[center_y+math.ceil(y_p)-1,center_x-math.floor(x_p)-1]        #...lower right pixel  

              #       quadrant II
              Output[center_y-k,center_x-l]=a1*b1*CCD[center_y-math.floor(y_p)-1,center_x-math.floor(x_p)-1]                                     #...lower right pixel
              Output[center_y-k,center_x-l]+=a2*b2*CCD[center_y-math.floor(y_p)-1,center_x-math.ceil(x_p)-1]      #...lower left pixel  
              Output[center_y-k,center_x-l]+=a3*b3*CCD[center_y-math.ceil(y_p)-1,center_x-math.ceil(x_p)-1]     #...upper left pixel  
              Output[center_y-k,center_x-l]+=a4*b4*CCD[center_y-math.ceil(y_p)-1,center_x-math.floor(x_p)-1]      #...upper right pixel  

              #       quadrant I
              Output[center_y-k,center_x+l]=a1*b1*CCD[center_y-math.floor(y_p)-1,center_x+math.floor(x_p)-1]                                     #...lower left pixel
              Output[center_y-k,center_x+l]+=a2*b2*CCD[center_y-math.floor(y_p)-1,center_x+math.ceil(x_p)-1]        #...lower right pixel 
              Output[center_y-k,center_x+l]+=a3*b3*CCD[center_y-math.ceil(y_p)-1,center_x+math.ceil(x_p)-1]       #...upper right pixel
              Output[center_y-k,center_x+l]+=a4*b4*CCD[center_y-math.ceil(y_p)-1,center_x+math.floor(x_p)-1]       #...upper left pixel


    return Output