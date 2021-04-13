"""
Python Dictionary for getting the data out of the files recorded with MAX P04

2021
@authors:   KG: Kathinka Gerlinger (kathinka.gerlinger@mbi-berlin.de)
            RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from lmfit.models import GaussianModel, ConstantModel
from scipy.stats import linregress

def load_scanfile(fname, INSTRUMENT_KEYS):
    '''
    Load scanfile
    INPUT:  fname: str, name of the scanfile
            INSTRUMENT_KEYS: dict, names of the data in the returned xarray and the string where to find it in the scanfile
    OUTPUT: xarray of the scandata
    --------
    author: MS 2021
    '''
    scandata = xr.Dataset()
    dimcount = 0
    with h5py.File(fname, 'r') as f:
        for k in f['scan/data'].keys():
            data = f['scan/data/' + k][()]
            dims = ['index']
            ndim = len(data.shape)
            for i in range(ndim - 1):
                dims.append(f'dim_{dimcount}')
                dimcount += 1
            scandata[k] = (dims, data)
        for k, v in INSTRUMENT_KEYS.items():
            scandata[k] = ('instrument', f[v][()])
    return scandata

def load_scanlist(scanlist, INSTRUMENT_KEYS):
    '''
    Load a list of compatible scans and concatenate them to a single dataset.
    INPUT:  scanlist: list of filenames
            INSTRUMENT_KEYS: dict, names of the data in the returned xarray and the string where to find it in the scanfile
    OUTPUT: xarray of the scandata
    --------
    author: MS 2021
    '''
    scans = []
    for s in scanlist:
        scans.append(load_scanfile(s, INSTRUMENT_KEYS))

    scan = xr.concat(scans, 'scan_nr')
    scan = scan.assign_coords({'scan_nr': scanlist})
    if 'keysight1_current' in scan:
        scan['keysight1_current'] = 1e9 * scan.keysight1_current
    return scan


def fit_samplescan(positions, intensity, weights=None, doublepeak=False):
    '''
    fit a gaussian to the intensity vs motorposition
    INPUT:  positions: array of motorpositions
            intensity: array of the intensities
            weights: optional, 
            douplepeak: optional, bool, wether to fit two peaks (default is False)
    OUTPUT: xarray of the scandata
    --------
    author: MS 2021
    '''
    span = positions.max() - positions.min()
    
    model = GaussianModel()
    guess = model.guess(intensity, x=positions, weights=weights)
    model += ConstantModel()
    param = model.make_params()
    param.update(guess)
    if doublepeak:
        model += GaussianModel(prefix='p2_')
        param2 = model.make_params()
        param2['sigma'].set(value=param['sigma'].value / 2)
        param2['p2_sigma'].set(value=param['sigma'].value / 2)
        param2['center'].set(value=param['center'].value - span / 10)
        param2['p2_center'].set(value=param['center'].value + span / 10)
        param2['amplitude'].set(value=param['amplitude'].value / 3, min=0)
        param2['p2_amplitude'].set(value=param['amplitude'].value / 3, min=0)
        param = param2
    fit = model.fit(intensity, x=positions, weights=weights, params=param)
    return fit


def plot_samplescans(scan, mov):
    '''
    Fit and plot the scans
    INPUT:  scan: xarray of loaded scans
            mov: str, name of the moved motor
    OUTPUT: xarray of the scandata with fits and figure
    --------
    author: MS 2021
    '''
    npts = len(scan_nr)
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, npts))
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    scan['centers'] = ('scan_nr', np.zeros(npts))
    scan['centers_err'] = ('scan_nr', np.zeros(npts))

    for i, s in enumerate(scan.scan_nr.values):
        sel = scan.sel(scan_nr=s)
        c = colors[i]
        weights = sel.weights if 'weights' in sel else None
        doublepeak = (mov == 'sz')
        fit = fit_samplescan(sel[mov].values, sel.keysight1_current.values,
                             weights, doublepeak)
        ax1.plot(sel[mov], sel.keysight1_current, '-', c=c, label=s, alpha=.4)
        ax1.plot(sel[mov], fit.eval(x=sel[mov]), '-', c=c, lw=.6)
        ax1.axvline(fit.params['center'].value, c=c)
        scan['centers'].loc[{'scan_nr': s}] = fit.params['center'].value
        scan['centers_err'].loc[{'scan_nr': s}] = fit.params['center'].stderr
    ax2.errorbar(
        scan.scan_nr,
        1e3 * (scan.centers - scan.centers.mean()),
        yerr=scan.centers_err,
        fmt='o-', ms=4)
    ax1.legend(fontsize=8, ncol=2)

    ax1.set_xlabel(mov)
    ax1.set_ylabel('diode current (nA)')
    ax2.set_xlabel('scan number')
    ax2.set_ylabel('$\Delta$ peak center (µm)')
    ax2.set_xticks(scan.scan_nr)
    ax2.grid(True)
    plt.tight_layout(pad=.3)
    return fig, scan