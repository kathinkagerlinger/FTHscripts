"""
Python Dictionary for FTH reconstructions

2016/2019
@author: dscran & KG
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import configparser as cp
import matplotlib.pyplot as plt
import pandas as pd
#reading matlap files
import scipy.io as sio
import scipy.constants as cst

from skimage.draw import circle

###########################################################################################

#                               LOAD DATA                                                 #

###########################################################################################

def load_both(pos, neg, auto_factor=False):
    '''
    Load images for a double helicity reconstruction
    INPUT:  pos, neg: images of positive and negative helicity
            auto_factor: determine the factor by which neg is multiplied automatically, if FALSE: factor is set to 1
    '''
    size = pos.shape
    if auto_factor:
        offset_pos = (np.mean(pos[:10,:10]) + np.mean(pos[-10:,:10]) + np.mean(pos[:10,-10:]) + np.mean(pos[-10:,-10:]))/4
        offset_neg = (np.mean(neg[:10,:10]) + np.mean(neg[-10:,:10]) + np.mean(neg[:10,-10:]) + np.mean(neg[-10:,-10:]))/4
        topo = pos - offset_pos + neg - offset_neg
        pos = pos - offset_pos
        factor = np.sum(np.multiply(pos - offset_pos,topo))/np.sum(np.multiply(topo, topo))
        print('Auto factor = ' + str(factor))
    else:
        topo = pos + neg
        factor = 1

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (pos[:, :size[0]] - factor * topo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (pos[:size[1], :] - factor * topo[:size[1], :], factor)
    else:
        return (pos - factor * topo, factor)

def load_single(image, topo, helicity, auto_factor=False):
    '''
    Load image for a single helicity reconstruction
    INPUT:  image: data of the single helicity image
            topo: topography data
            helicity: True/False or 1/0 for pos/neg helicity image
            auto_factor: determine the factor by which the topology is multiplied automatically, if FALSE: factor is set to 0.5
    '''
    #load the reference for topology
    topo = topo.astype(np.int64)
    #load the single image
    image = image.astype(np.int64)

    size = image.shape

    if auto_factor:
        offset_sing = (np.mean(image[:10,:10]) + np.mean(image[-10:,:10]) + np.mean(image[:10,-10:]) + np.mean(image[-10:,-10:]))/4
        offset_topo = (np.mean(topo[:10,:10]) + np.mean(topo[-10:,:10]) + np.mean(topo[:10,-10:]) + np.mean(topo[-10:,-10:]))/4
        factor = np.sum(np.multiply(image - offset_sing,topo - offset_topo))/np.sum(np.multiply(topo - offset_topo, topo - offset_topo))
        print('Auto factor = ' + str(factor))
    else:
        factor = 0.5

    if helicity:
        holo = image - factor * topo
    else:
        holo = -1 * (image - factor * topo)

    #make sure to return a quadratic image, otherwise the fft will distort the image
    if size[0]<size[1]:
        return (holo[:, :size[0]], factor)
    elif size[0]>size[1]:
        return (holo[:size[1], :], factor)
    else:
        return (holo, factor)



def load_mat(folder, npos):
    '''
    load the reconstruction file from the matlab routine
    '''
    rec_params = sio.loadmat(folder + 'holo_%04d.mat'%npos)
    center = rec_params['middle'][0]
    beamstop = rec_params['bmask']
    bs_diam = np.max(np.append(np.sum(beamstop, axis=1), np.sum(beamstop, axis=0)))
    print('Loaded matlab file ' + folder + 'holo_%04d.mat'%npos)
    return (center, bs_diam)

###########################################################################################

#                               PLOTTING                                                  #

###########################################################################################

def plot(image, scale = (2,98), color = 'gray', colorbar = True):
    '''
    plot the image with the given scale, colormap and with a colorbar
    '''
    mi, ma = np.percentile(image, scale)
    fig, ax = plt.subplots()
    im = ax.imshow(image, vmin = mi, vmax = ma, cmap = color)
    if colorbar:
        plt.colorbar(im)
    return (fig, ax)

def plot_ROI(image, ROI_coord, scale = (0,100), color = 'gray', colorbar = True):
    '''
    Plot the ROI of the image
    '''
    mi, ma = np.percentile(image[ROI_coord[2]:ROI_coord[3], ROI_coord[0]:ROI_coord[1]], scale)
    fig, ax = plt.subplots()
    ax = plt.imshow(np.real(image[ROI_coord[2]:ROI_coord[3], ROI_coord[0]:ROI_coord[1]]), vmin = mi, vmax = ma, cmap = color)
    if colorbar:
        plt.colorbar()
    return

###########################################################################################

#                               COSMIC RAYS                                               #

###########################################################################################
 
def remove_cosmic_ray(holo, coordinates):
    x = coordinates[0]
    y = coordinates[1]
    avg = 0
    for i in (x-1, x, x+1):
        for j in (y-1, y, y+1):
            if not np.logical_and(i == x, j == y):
                avg += holo[j, i]
    holo[y, x] = avg/8
    return holo

def remove_two(holo, x_coord, y_coord):
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)
    try:
        if x_coord.shape[0] == 2:
            holo = fth.remove_cosmic_ray(holo, [x_coord[0], y_coord])
            holo = fth.remove_cosmic_ray(holo, [x_coord[1], y_coord])
            holo = fth.remove_cosmic_ray(holo, [x_coord[0], y_coord])
    except:
        try:
            if y_coord.shape[0] == 2:
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[0]])
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[1]])
                holo = fth.remove_cosmic_ray(holo, [x_coord, y_coord[0]])
        except:
            print("No cosmic rays removed! Input two pixel!")
    return holo
    

###########################################################################################

#                               RECONSTRUCTION                                            #

###########################################################################################

def reconstruct(image):
    #Reconstruct the image by fft
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image)))


###########################################################################################

#                               CENTERING                                                 #

###########################################################################################

def integer(n):
    return np.int(np.round(n))

def set_center(image, center):
    '''this centering routine shifts the image in a cyclical fashion'''
    xdim, ydim = image.shape
    xshift = integer(xdim / 2 - center[1])
    yshift = integer(ydim / 2 - center[0])
    image_shift = np.roll(image, yshift, axis=0)
    image_shift = np.roll(image_shift, xshift, axis=1)
    print('Shifted image by %i pixels in x and %i pixels in y.'%(xshift, yshift))
    return image_shift


###########################################################################################

#                                 BEAM STOP MASK                                          #

###########################################################################################

def mask_beamstop(image, bs_size, sigma=10, center = None):
    '''A smoothed circular region of the imput image is set to zero.
    bs_size is diameter of the beamstop!'''

    #Save the center of the beamstop. If none is given, take the center of the image.
    if center is None:
        x0, y0 = [integer(c/2) for c in image.shape]
    else:
        x0, y0 = [integer(c) for c in center]

    #create the beamstop mask using scikit-image's circle function
    bs_mask = np.zeros(image.shape)
    yy, xx = circle(y0, x0, bs_size/2)
    bs_mask[yy, xx] = 1
    bs_mask = np.logical_not(bs_mask).astype(np.float64)
    #smooth the mask with a gaussion filter    
    bs_mask = gaussian_filter(bs_mask, sigma, mode='constant', cval=1)
    return image*bs_mask

def mask_beamstop_matlab(image, mask, sigma=8):
    '''If a binary mask the size of the image is given, use this function
    '''
    if np.logical_not(mask[0,0]): #if the outside is 0 and the beamstop is one
        mask = np.logical_not(mask).astype(np.float64)
    bs_mask = gaussian_filter(mask, sigma, mode='constant', cval=1)
    return image*bs_mask

###########################################################################################

#                                 PROPAGATION                                             #

###########################################################################################

def propagate(holo, prop_l, ccd_dist=18e-2, energy=779.5, integer_wl_multiple=True, px_size=13.5e-6):
    '''
    Parameters:
    ===========
    holo : hologram  to be propagated
    prop_l : propagation distance [m]
    ccd_dist : CCD - sample distance [m]
    energy : photon energy [eV] 
    integer_wl_mult : if true, coerce propagation distance to nearest integermultiple of photon wavelength 
    
    Returns:
    ========
    holo : propagated hologram
    
    ========
    written by Michael Schneider 2016
    '''
    wl = cst.h * cst.c / (energy * cst.e)
    if integer_wl_multiple:
        prop_l = np.round(prop_l / wl) * wl

    l1, l2 = holo.shape
    q0, p0 = [s / 2 for s in holo.shape] # centre of the hologram
    q, p = np.mgrid[0:l1, 0:l2]  #grid over CCD pixel coordinates   
    pq_grid = (q - q0) ** 2 + (p - p0) ** 2 #grid over CCD pixel coordinates, (0,0) is the centre position
    dist_wl = 2 * prop_l * np.pi / wl
    phase = (dist_wl * np.sqrt(1 - (px_size / ccd_dist) ** 2 * pq_grid))
    holo = np.exp(1j * phase) * holo

    print ('Propagation distance: %.2fum' % (prop_l*1e6)) 
    return holo

###########################################################################################

#                                   PHASE SHIFTS                                          #

###########################################################################################

def global_phase_shift(holo, phi):
    #multiply the hologram with a global phase
    return holo*np.exp(1j*phi)

###########################################################################################

#                                  CONFIG FILES                                           #

###########################################################################################

def save_hdf(image_numbers, center_coordinates, bs_size, prop_dist, phase_shift, roi_coordinates, filename, key, topo = None):
    '''save the reconstruction parameters in a hdf file
    PARAMETERS:
        image_numbers is a 1D list with either two values: [pos, neg], in case of single helicity, one of these images should be NaN
        center_coordinates is a 1D array with two values: [xcenter, ycenter]
        roi_coordinates is a 1D array with four calues: [xstart, xstop, ystart, ystop]
        bs_size is a float indicating the diameter of the beamstop
        prop_dist is a float indicating the propagation length
        filename is the name and path under which the configfile should be saved'''
    if topo is None:
        topo = [np.nan, np.nan]
    #create the data frame
    df = pd.DataFrame({'Image Number pos': image_numbers[0], 'Image Number neg': image_numbers[1], 'Topo Number pos': topo[0], 'Topo Number pos': topo[1], 'Center x': center_coordinates[0],
                       'Center y': center_coordinates[1], 'beamstop diameter': bs_size, 'propagation dist.': prop_dist, 'Phase': phase_shift, 'ROI x1': roi_coordinates[0],
                       'ROI x2': roi_coordinates[1], 'ROI y1': roi_coordinates[2], 'ROI y2': roi_coordinates[3]})
    #write the file
    print('Save HDF file ' + filename)
    df.to_hdf(filename, key)
    return

def save_config(image_numbers, center_coordinates, bs_size, prop_dist, phase_shift, roi_coordinates, conf_filename):
    '''save the reconstruction parameters in a config file with configparser
    PARAMETERS:
        image_numbers is a 1D list with either one or two values: [single_hel_number] or [pos, neg]
        center_coordinates is a 1D array with two values: [xcenter, ycenter]
        roi_coordinates is a 1D array with four calues: [xstart, xstop, ystart, ystop]
        bs_size is a float indicating the diameter of the beamstop
        prop_dist is a float indicating the propagation length
        conf_filename is the name and path under which the configfile should be saved'''
    def list_to_str(mylist): #configparser can only read and write strings
        str_list = str()
        for i in mylist:
            str_list += str(i) + ' '
        return str_list

    #create the config data
    config = cp.ConfigParser()
    config['Image']={}
    config['Image']['Numbers'] = list_to_str(image_numbers)
    config['Center'] = {}
    config['Center']['Coordinates'] = list_to_str(center_coordinates)
    config['Beamstop'] = {}
    config['Beamstop']['Size'] = str(bs_size)
    config['Propagation'] = {}
    config['Propagation']['Distance'] = str(prop_dist)
    config['Phase Shift'] = {}
    config['Phase Shift']['phase'] = str(phase_shift)
    config['ROI'] = {}
    config['ROI']['Coordinates'] = list_to_str(roi_coordinates)

    print('Save Config file ' + conf_filename)
    #write the file
    with open(conf_filename, 'w') as configfile:
        config.write(configfile)
    return

def save_config_matlab(image_numbers, center_coordinates,  prop_dist, phase_shift, roi_coordinates, conf_filename):
    '''save the reconstruction parameters in a config file with configparser
    PARAMETERS:
        image_numbers is a 1D list with either one or two values: [single_hel_number] or [pos, neg]
        center_coordinates is a 1D array with two values: [xcenter, ycenter]
        prop_dist is a float indicating the propagation length
        phase_shift is a float indicating the global phase shift
        roi_coordinates is a 1D array with four calues: [xstart, xstop, ystart, ystop]
        conf_filename is the name and path under which the configfile should be saved'''
    def list_to_str(mylist): #configparser can only read and write strings
        str_list = str()
        for i in mylist:
            str_list += str(i) + ' '
        return str_list

    #create the config data
    config = cp.ConfigParser()
    config['Image']={}
    config['Image']['Numbers'] = list_to_str(image_numbers)
    config['Center'] = {}
    config['Center']['Coordinates'] = list_to_str(center_coordinates)
    config['Propagation'] = {}
    config['Propagation']['Distance'] = str(prop_dist)
    config['Phase Shift'] = {}
    config['Phase Shift']['phase'] = str(phase_shift)
    config['ROI'] = {}
    config['ROI']['Coordinates'] = list_to_str(roi_coordinates)

    print('Save Config file ' + conf_filename)
    #write the file
    with open(conf_filename, 'w') as configfile:
        config.write(configfile)
    return

def read_hdf(filename, key):
    '''read data from hdf file 
    filename is the name and path under which the hdf is saved, key is the key or which part of the file you want to get'''
    df = pd.read_hdf(filename, key)
    
    if np.isnan(df['Image Number pos']):
        image_numbers = np.array([np.nan, int(df['Image Number neg'])])
    elif np.isnan(df['Image Number neg']):
        image_numbers = np.array([int(df['Image Number pos']), np.nan])
    else:
        image_numbers = np.array([int(df['Image Number pos']), int(df['Image Number neg'])])
    
    center = np.array([int(df['Center x']), int(df['Center y'])])
    roi = np.array([int(df['ROI x1']), int(df['ROI x2']), int(df['ROI y1']), int(df['ROI y2'])])
    
    return (image_numbers, center, int(df['beamstop diameter']), int(df['propagation dist.']), int(df['Phase']), roi)


def read_config(conf_filename):
    '''read data from config file created with configparser
    conf_filename is the name and path under which the configfile is saved'''
    def str_to_list(mystr):#configparser can only read and write strings
        def append_string(a,b):
            if np.isnan(b):
                a.append(np.nan)
            else:
                a.append(int(b))
            return a

        mylist = []
        tmp = str()
        for i in mystr:
            if i == ' ':
                mylist = append_string(mylist, float(tmp))
                tmp=str()
            else:
                tmp += i
        mylist = append_string(mylist, float(tmp))
        return mylist

    print('Read Config file ' + conf_filename)
    #read the config file
    conf= cp.ConfigParser()
    conf.read(conf_filename)

    #save the parameters
    image_numbers = str_to_list(conf['Image']['Numbers'])
    center = str_to_list(conf['Center']['Coordinates'])
    bs_diam = np.float(conf['Beamstop']['Size'])
    prop_dist = np.float(conf['Propagation']['Distance'])
    phase = np.float(conf['Phase Shift']['phase'])
    roi = str_to_list(conf['ROI']['Coordinates'])

    return (image_numbers, center, bs_diam, prop_dist, phase, roi)

def read_config_matlab(conf_filename):
    '''read data from config file created with configparser
    conf_filename is the name and path under which the configfile is saved'''
    def str_to_list(mystr):#configparser can only read and write strings
        def append_string(a,b):
            if np.isnan(b):
                a.append(np.nan)
            else:
                a.append(int(b))
            return a

        mylist = []
        tmp = str()
        for i in mystr:
            if i == ' ':
                mylist = append_string(mylist, float(tmp))
                print(mylist)
                tmp=str()
            else:
                tmp += i
        mylist = append_string(mylist, float(tmp))
        return mylist

    print('Read Config file ' + conf_filename)
    #read the config file
    conf= cp.ConfigParser()
    conf.read(conf_filename)

    #save the parameters
    image_numbers = str_to_list(conf['Image']['Numbers'])
    center = str_to_list(conf['Center']['Coordinates'])
    prop_dist = np.float(conf['Propagation']['Distance'])
    phase = np.float(conf['Phase Shift']['phase'])
    roi = str_to_list(conf['ROI']['Coordinates'])


    return (image_numbers, center, prop_dist, phase, roi)