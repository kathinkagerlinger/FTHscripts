"""
Python Dictionary for FTH reconstructions in Python using functions defined in fth_reconstroction

2019
@author: KG
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
#interactive stuff
import ipywidgets as widgets
from IPython.display import display
#fth dictionary
import fth_reconstruction as fth
import cameras as cam





###########################################################################################

#                         RECONSTRUCT WITH MATLAB FILE                                    #

###########################################################################################


def allNew(image_folder, image_numbers, folder_matlab, matlab_number = None, size=[2052,2046], auto_factor = True, spe_prefix = None):
    '''
    opens the image files in image_numbers (first element is POSITIVE HEL, second is NEGATIVE HEL)
    if a spe_prefix is given, it opens spe files, otherwise it opens greateyes files.
    opens the matlab file for the center and beamstop

    shifts the center and masks the beamstop
    returns the shifted and masked hologram, the center and the beamstop
    '''
    if spe_prefix is None:
        pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[0], size=size)
        neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[1], size=size)
    else: 
        pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[0], return_header=False)
        neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[1], return_header=False)

    holo, factor = fth.load_both(pos, neg, auto_factor = auto_factor)
    if matlab_number is None:
        matlab_number = image_numbers[0]
    center, bs_diam = fth.load_mat(folder_matlab, matlab_number)
    
    print("Start reconstructing the image using the center from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)
    #holoN = fth.mask_beamstop(holoN, beamstop, sigma=10)

    print("Please adapt the beamstop executing the function set_beamstop.")
    print("Please chose a ROI by executing the function set_roi.")
    return (holoN, center, bs_diam, factor)


def change_beamstop(bs_diameter):
    '''
    Change the beamstop diameter with an input field.
    Returns the input field. When you are finished, you can save the positions of the field.
    '''
    style = {'description_width': 'initial'}

    bs_input = widgets.IntText(value=bs_diameter, description='ROI x1 coordinate:', disabled=False, style=style)

    button = widgets.Button(description='Finished')
    display(bs_input, button)

    return (bs_input, button)

def set_beamstop(holo, bs_diameter, sigma = 10):
    '''
    Input center-shifted hologram and the diameter of the beamstop. You may change the sigma of the gauss filter (default is 10).
    Returns the hologram where the beamstop is masked.
    '''
    print("Masking beamstop with a diameter of %i pixels."%bs_diameter)
    holoB = fth.mask_beamstop(holo, bs_diameter, sigma=sigma)
    return holoB


def set_roi(holo, scale = (1,99)):
    """ 
    Select a ROI somewhat interactively
    Input the shfited and masked hologram as returned from recon_allNew.
    Returns the four input fields. When you are finished, you can save the positions of the fields.
    """
    recon=fth.reconstruct(holo)
    
    mi, ma = np.percentile(np.real(recon), scale)
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(recon), cmap='gray', vmin=mi, vmax=ma)
    plt.colorbar()

    style = {'description_width': 'initial'}

    ROIx1 = widgets.IntText(value=None, description='ROI x1 coordinate:', disabled=False, style=style)
    ROIx2 = widgets.IntText(value=None, description='ROI x2 coordinate:', disabled=False, style=style)
    ROIy1 = widgets.IntText(value=None, description='ROI y1 coordinate:', disabled=False, style=style)
    ROIy2 = widgets.IntText(value=None, description='ROI y2 coordinate:', disabled=False, style=style)

    button = widgets.Button(description='Finished')
    display(ROIx1, ROIx2, ROIy1, ROIy2, button)

    return (ROIx1, ROIx2, ROIy1, ROIy2, button)


def plot_ROI(holo, ROI_coord):
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(fth.reconstruct(holo)[ROI_coord[2]:ROI_coord[3], ROI_coord[0]:ROI_coord[1]]), cmap='gray')
    return


def propagate(holo, ROI, phase=0, prop_dist=0, scale=(0,100)):
    '''
    starts the quest for the right propagation distance and global phase shift.
    Input the shfited and masked hologram as returned from recon_allNew as well as the determined coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
    Returns the two sliders. When you are finished, you can save the positions of the sliders.
    '''
    style = {'description_width': 'initial'}
    fig, axs = plt.subplots(1,2)
    def p(x,y):
        image = fth.reconstruct(fth.propagate(holo, x*1e-6)*np.exp(1j*y))
        mir, mar = np.percentile(np.real(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)
        mii, mai = np.percentile(np.imag(image[ROI[2]:ROI[3], ROI[0]:ROI[1]]), scale)

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

    return (slider_prop, slider_phase, button)



###########################################################################################

#                              RECONSTRUCT WITH CONFIG FILE                               #

###########################################################################################

def fromParameters(image_folder, image_numbers, fname_param, new_bs=False, old_prop=True, topo_nr=None, helpos=None, auto_factor=False, size=[2052,2046], spe_prefix=None):
    '''
    opens the image files in image_numbers (either single helicity or double helicity)
    if a spe_prefix is given, it opens spe files, otherwise it opens greateyes files.
    opens the config file for the center, beamstop, the ROI and the propagation etc.

    shifts the center and masks the beamstop
    returns the reconstruction parameters as well as the hologram that was corrected as inidcated (with and without bs mask, propagation)
    '''

    #Load the parameters from the hdf file
    _, _, _, center, bs_diam, prop_dist, phase, roi = fth.read_hdf(fname_param)

    #Load the images (spe or greateyes; single or double helicity)
    if spe_prefix is None: #greateyes
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[0], size=size)
            neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[1], size=size)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[0], size=size)
                    neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[1], size=size)
            else:
                pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[0], size=size)
                neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[1], size=size)
    
            image = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers, size=size)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    else: #spe
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[0], return_header=False)
            neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[1], return_header=False)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[0], return_header=False)
                    neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[1], return_header=False)
            else:
                pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[0], return_header=False)
                neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[1], return_header=False)
    
            image = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers, return_header=False)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    
    print("Start reconstructing the image using the center and beamstop mask from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)

    if not new_bs:
        print("Using beamstop diameter %i from config file and a sigma of 10."%bs_diam)
        holoN = fth.mask_beamstop(holoN, bs_diam, sigma=10)
    else:
        print("Please adapt the beamstop using the beamstop function and then propagate.")
        return(holoN, factor, center, bs_diam, roi, prop_dist, phase)
        
    if old_prop:
        print("Using propagation distance from config file.")
        holoN = fth.propagate(holoN, prop_dist*1e-6)
        print("Now determine the global phase shift by executing phase_shift.")
    else: 
        print("Please use the propagation function to propagate.")

    return(holoN, factor, center, bs_diam, roi, prop_dist, phase)



def fromConfig(image_folder, image_numbers, folder_config, number_config, new_bs=False, old_prop=True, topo_nr=None, helpos=None, auto_factor=False, size=[2052,2046], spe_prefix=None):
    '''
    opens the image files in image_numbers (either single helicity or double helicity)
    if a spe_prefix is given, it opens spe files, otherwise it opens greateyes files.
    opens the config file for the center, beamstop, the ROI and the propagation etc.

    shifts the center and masks the beamstop
    returns the reconstruction parameters as well as the hologram that was corrected as inidcated (with and without bs mask, propagation)
    '''

    #Load the config file
    nref, center, bs_diam, prop_dist, phase, roi = fth.read_config(folder_config + '%i_config.ini'%number_config)

    #Load the images (spe or greateyes; single or double helicity)
    if spe_prefix is None: #greateyes
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[0], size=size)
            neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers[1], size=size)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[0], size=size)
                    neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%nref[1], size=size)
            else:
                pos = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[0], size=size)
                neg = cam.load_greateyes(image_folder + 'holo_%04d.dat'%topo_nr[1], size=size)
    
            image = cam.load_greateyes(image_folder + 'holo_%04d.dat'%image_numbers, size=size)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    else: #spe
        if helpos==None:
            print("Double Helicity Reconstruction")
            pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[0], return_header=False)
            neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers[1], return_header=False)
            holo, factor = fth.load_both(pos, neg, auto_factor=auto_factor)
        else:
            print("Single Helicity Reconstruction")
            if topo_nr is None:
                if np.logical_or(np.isnan(nref[0]), np.isnan(nref[0])):
                    print("Please put in the numbers for the topography.")
                    return
                else:
                    pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[0], return_header=False)
                    neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%n_ref[1], return_header=False)
            else:
                pos = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[0], return_header=False)
                neg = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%topo_nr[1], return_header=False)
    
            image = cam.load_spe(image_folder + spe_prefix + '%04d.spe'%image_numbers, return_header=False)
            holo, factor = fth.load_single(image, pos+neg, helpos, auto_factor=auto_factor)
    
    print("Start reconstructing the image using the center and beamstop mask from the Matlab reconstruction.")
    holoN = fth.set_center(holo, center)

    if not new_bs:
        print("Using beamstop diameter %i from config file and a sigma of 10."%bs_diam)
        holoN = fth.mask_beamstop(holoN, bs_diam, sigma=10)
    else:
        print("Please adapt the beamstop using the beamstop function and then propagate.")
        return(holoN, center, bs_diam, roi, prop_dist, phase)
        
    if old_prop:
        print("Using propagation distance from config file.")
        holoN = fth.propagate(holoN, prop_dist*1e-6)
        print("Now determine the global phase shift by executing phase_shift.")
    else: 
        print("Please use the propagation function to propagate.")

    return(holoN, factor, center, bs_diam, roi, prop_dist, phase)


def phase_shift(holo, roi, phase=0):
    '''
    starts the quest for the global phase shift.
    Input the shfited, masked and propagated hologram as returned from recon_fromConfig as well as the determined coordinates of the ROI in the order [Xstart, Xstop, Ystart, Ystop]
    Returns the slider. When you are finished, you can save the positions of the slider.
    '''
    fig, axs = plt.subplots(1,2)
    def p(x):
        image = fth.reconstruct(holo*np.exp(1j*x))
        ax1 = axs[0].imshow(np.real(image[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        #fig.colorbar(ax1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title("Real Part")
        ax2 = axs[1].imshow(np.imag(image[roi[2]:roi[3], roi[0]:roi[1]]), cmap='gray')
        #fig.colorbar(ax2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title("Imaginary Part")
        fig.tight_layout()
        print('REAL: max=%i, min=%i'%(np.max(np.real(image)), np.min(np.real(image))))
        print('IMAG: max=%i, min=%i'%(np.max(np.imag(image)), np.min(np.imag(image))))
        return

    layout = widgets.Layout(width='90%')
    style = {'description_width': 'initial'}
    slider_phase = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.001, value=phase, layout=layout, description='phase shift', style=style)

    widgets.interact(p, x=slider_phase)

    #input("Press the <ENTER> key to continue...")
    button = widgets.Button(description="Finished")
    display(button)
    
    def on_button_clicked(b):
        slider_phase.close()
        return
    button.on_click(on_button_clicked)
    return (slider_phase, button)



###########################################################################################

#                               SAVE THE CONFIG                                           #

###########################################################################################


def save_parameters(fname, recon, factor, center, bs_diam, prop_dist, phase, roi, image_numbers, comment = '', topo = None):
    '''
    reconstruct the shifted and masked hologram (propagation and phase shift are performed here.)
    save all parameters in numpy-files (holo and beamstop) and a config file (rest)
    if the folder you put in does not exist, it will be created.
    '''
    image_numbers = np.array(image_numbers)
    
    if image_numbers.size == 1:
        im = image_numbers
    elif np.isnan(image_numbers[0]):
        im = image_numbers[1]
    else:
        im = image_numbers[0]
    
    if topo is None:
        topo = [np.nan, np.nan]
    
    reco_dict = {
        'reconstruction': recon,
        'image numbers': image_numbers,
        'topo numbers': topo,
        'factor': factor,
        'center': center,
        'beamstop diameter': bs_diam,
        'ROI coordinates': roi,
        'Propagation distance': prop_dist,
        'phase': phase,
        'comment': comment
    }
    
    fth.save_reco_dict_to_hdf(fname, reco_dict)
    return


def save_parameters_config(holo, center, prop_dist, phase, roi, folder, image_numbers, bs_diam, propagate=False):
    '''
    reconstruct the shifted and masked hologram (propagation and phase shift are performed here.)
    save all parameters in numpy-files (holo and beamstop) and a config file (rest)
    if the folder you put in does not exist, it will be created.
    '''
    image_numbers = np.array(image_numbers)
    if not(os.path.exists(folder)):
        print("Creating folder " + folder)
        os.mkdir(folder)

    if propagate:
        recon = fth.reconstruct(fth.propagate(holo, prop_dist*1e-6)*np.exp(1j*phase))
    else:
        recon = fth.reconstruct(holo*np.exp(1j*phase))
    print('Shifted phase by %f.'%phase)
    
    if image_numbers.size == 1:
        im = image_numbers
    elif np.isnan(image_numbers[0]):
        im = image_numbers[1]
    else:
        im = image_numbers[0]

    np.save(folder + '%i_recon'%im, recon[roi[2]:roi[3], roi[0]:roi[1]])
    
    fth.save_config(image_numbers, center, bs_diam, prop_dist, phase, roi, folder + '%i_config.ini'%im)
    return