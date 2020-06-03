import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
import fth_reconstruction as fth_rec
from IPython.display import display
from IPython.display import clear_output

from skimage.draw import circle


#############################################################
#       PHASE RETRIEVAL FUNCTIONS
#############################################################

def PhaseRtrv(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,
       plot_every=20,ROI=[None,None,None,None],BS=[0,0,0],real_object=False,average_img=10):
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
            ax2.imshow(np.abs(np.fft.ifftshift(guess)), cmap='RdBu')#,interpolation='spline16')
            ax3.imshow(im_abs, cmap='binary')#,interpolation='spline16')

            abs_detail=im_abs[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            angle_detail=im_angle[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            ax4.imshow(abs_detail, cmap='binary')#,interpolation='spline16')
            ax5.imshow(angle_detail, cmap='RdBu',vmin=-np.pi/2,vmax=np.pi/2)#,interpolation='spline16')

            display(plt.gcf())
        
            print('step:',s,'   beta=',beta,'   alpha=',alpha, '    mode=',mode,
                  '    beta_mode=',beta_mode,'   N_avg=',average_img)

    #sum best guess images
    guess==np.sum(Best_guess,axis=0)/average_img

    #return final image
    if BS_radius==0:
        return np.fft.ifftshift(np.fft.fft2(diffract * np.exp(1j * np.angle(guess))))
    else:
        return np.fft.ifftshift(np.fft.fft2((1-BSmask) *diffract* np.exp(1j * np.angle(guess)) + guess*BSmask))
    

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
            ax2.imshow(np.abs(np.fft.ifftshift(guess)), cmap='RdBu')#,interpolation='spline16')
            ax3.imshow(im_abs, cmap='binary')#,interpolation='spline16')

            abs_detail=im_abs[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            angle_detail=im_angle[ROI[2]:ROI[3],ROI[0]:ROI[1]]
            ax4.imshow(abs_detail, cmap='binary')#,interpolation='spline16')
            ax5.imshow(angle_detail, cmap='RdBu',vmin=-np.pi/2,vmax=np.pi/2)#,interpolation='spline16')

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
        global BS,BS2,plot_every,mode,beta_mode,real_object,average,Nit,beta_zero
        if BeamStop==False:
            BS2=[0,0,0]
        else:
            BS2=BS
        plot_every=plot_how_often
        mode = algorithm
        beta_mode=beta_func
        real_object = only_real
        average=N_average_images
        Nit=N_step
        beta_zero=beta
        return (Nit,mode,beta_mode,real_object,plot_every,BS2,average,beta_zero)
        
    widgets.interact(f, N_step=widgets.IntSlider(min=0, max=3000, step=5, value=200),
        algorithm=['ER','RAAR','HIO','CHIO','HIOs','OSS','HPR'],
         beta=widgets.FloatSlider(value=0.8,min=0.5,max=1.0,step=0.01),
         beta_func=['const','arctan','exp','linear_to_beta_zero','linear_to_1'],
         only_real=False, BeamStop=False,
         N_average_images=widgets.IntSlider(min=1, max=30, step=1, value=10),
         plot_how_often=widgets.IntSlider(min=5, max=300, step=5, value=50));
    
    return (Nit,mode,beta_mode,real_object,plot_every,BS2,average,beta_zero)
