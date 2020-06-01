"""
Python Dictionary for getting the data out of the hdf files recorded with MAXI

2020
@author: dscran, KG, RB
"""
import h5py
import numpy as np

def measurement_info(fname, entry_number):
    '''
    Prints all the keys in the measurement.
    INPUT:  fname: path and name of the hdf file
            entry_number: number of the entry you want to check
    OUTPUT: None
    -----
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    print(list(f[f'entry{entry_number:d}/measurement'].keys()))
    return


def diode_scan(fname, entry_number, motor):
    '''
    Function to evaluate a diode scan.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
            motor: string, name of the motor used in the diode scan. You can check it with measurement_info()
    OUTPUT: diode: list of the values measured by the diode
            motor_val: list of the motorpositions
    -----
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    diode = f[f'entry{entry_number:d}/measurement/diodeA'][()]
    motor_val = f[f'entry{entry_number:d}/measurement/{motor:s}'][()]
    return (diode, motor_val)


def get_measurement(fname, entry_number, name):
    '''
    Function to get data of a specific entry and name, saved under measurement.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
            name: string, name of the key you want to read. You can check it with measurement_info()
    OUTPUT: list of the measured values
    -----
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    return f[f'entry{entry_number:d}/measurement/{name:s}'][()]


def get_mte(fname, entry_number):
    '''
    Function to get data of a the camera of a specific entry.
    INPUT:  fname: string, path and name of the hdf file
            entry_number: integer, number of the entry you want to check
    OUTPUT: array of the image data
    -----
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    return f[f'entry{entry_number:d}/measurement/mte'][()][0]