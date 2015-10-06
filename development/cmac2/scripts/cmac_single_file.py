"""
===============================================
Correct one file and save the results and plots
===============================================
Take one file at C or X band and retrieve texture
gate ID, dealias and process phase

"""

#Author: Scott Collis (scollis@anl.gov)
#Co-Author: Jonathan Helmus
#Credit: Py-ART, Scott Giangrande for original LP idea, Kai Muehlbauer for CyLP optimization

import pyart
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import ndimage, signal, integrate
import time
import copy
import netCDF4
import skfuzzy as fuzz
import argparse


def cum_score_fuzzy_logic(radar, mbfs = None,
                          debug = False, ret_scores = False,
                          hard_const = None):
    if mbfs == None:
        second_trip = {'velocity_texture' : [[0,0,80,90], 1.0],
                       'cross_correlation_ratio' : [[.5,.7,1,1], 0.0],
                       'normalized_coherent_power' : [[0,0,.5,.6], 3.0],
                       'height': [[0,0,5000,8000], 1.0],
                       'sounding_temperature' : [[-100,-100,100,100], 0.0],
                       'SNR' : [[15,20, 1000,1000],1.0]}

        rain = {'differential_phase_texture' : [[0,0,80,90], 1.0],
                       'cross_correlation_ratio' : [[0.94,0.96,1,1], 1.0],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 1.0],
                       'height': [[0,0,5000,6000], 0.0],
                       'sounding_temperature' : [[0,3,100,100], 2.0],
                       'SNR' : [[8,10, 1000,1000], 1.0]}

        snow = {'differential_phase_texture' : [[0,0,80,90], 1.0],
                       'cross_correlation_ratio' : [[0.85,0.9,1,1], 1.0],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 1.0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-100,-100,0,1.], 2.0],
                       'SNR' : [[8,10, 1000,1000], 1.0]}

        no_scatter = {'differential_phase_texture' : [[90,90,400,400], 0.0],
                       'cross_correlation_ratio' : [[0,0,0.1,0.2], 0.0],
                       'normalized_coherent_power' : [[0,0,0.1,0.2], 0.0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-100,-100,100,100], 0.0],
                       'SNR' : [[-100,-100, 8,10], 6.0]}

        melting = {'differential_phase_texture' : [[20,30,80,90], 0.0],
                       'cross_correlation_ratio' : [[0.6,0.7,.94,.96], 4.],
                       'normalized_coherent_power' : [[0.4,0.5,1,1], 0],
                       'height': [[0,0,25000,25000], 0.0],
                       'sounding_temperature' : [[-1.,0,3.5,5], 2.],
                       'SNR' : [[8,10, 1000,1000], 0.0]}

        mbfs = {'multi_trip': second_trip, 'rain' : rain,
                'snow' :snow, 'no_scatter' : no_scatter, 'melting' : melting}
    flds = radar.fields
    scores = {}
    for key in mbfs.keys():
        if debug: print('Doing ' + key)
        this_score = np.zeros(flds[flds.keys()[0]]['data'].shape).flatten() * 0.0
        for MBF in mbfs[key].keys():
            this_score = fuzz.trapmf(flds[MBF]['data'].flatten(),
                                     mbfs[key][MBF][0] )*mbfs[key][MBF][1] + this_score

        this_score = this_score.reshape(flds[flds.keys()[0]]['data'].shape)
        scores.update({key: ndimage.filters.median_filter(this_score, size = [3,4])})
    if hard_const != None:
        # hard_const = [[class, field, (v1, v2)], ...]
        for this_const in hard_const:
            if debug: print('Doing hard constraining ', this_const[0])
            key = this_const[0]
            const = this_const[1]
            fld_data = radar.fields[const]['data']
            lower = this_const[2][0]
            upper = this_const[2][1]
            const_area = np.where(np.logical_and(fld_data >= lower, fld_data <= upper))
            if debug: print(const_area)
            scores[key][const_area] = 0.0
    stacked_scores = np.dstack([scores[key] for key in scores.keys() ])
    #sum_of_scores = stacked_scores.sum(axis = 2)
    #print(sum_of_scores.shape)
    #norm_stacked_scores = stacked_scores
    max_score = stacked_scores.argmax(axis = 2)

    gid = {}
    gid['data'] = max_score
    gid['units'] = ''
    gid['standard_name'] = 'gate_id'

    strgs = ''
    i=0
    for key in scores.keys():
        strgs = strgs + str(i) + ': ' + key + ' '

    gid['long_name'] = 'Classification of dominant scatterer'
    gid['notes'] = strgs
    gid['valid_max'] = max_score.max()
    gid['valid_min'] = 0.0
    if ret_scores == False:
        rv = (gid, scores.keys())
    else:
        rv = (gid, scores.keys(), scores)
    return rv


if __name__ == "__main__":
    print("executing")
    version = '2.0E'
    parser = argparse.ArgumentParser(description='CMAC '+version)
    parser.add_argument("filename")
    parser.add_argument("isonde")
    parser.add_argument("--ngates",
            help="set to the number of gates you want to process, handy if you want to trim. Leave unset for all gates")
    args = parser.parse_args()
    print("Called with Filename: "+ args.filename +" isonde: " +
            args.isonde)
    radar = pyart.io.read(args.filename)
    if args.ngates != None:
        i_end = int(args.ngates)
        radar.range['data']=radar.range['data'][0:i_end]
        for key in radar.fields.keys():
            radar.fields[key]['data']= radar.fields[key]['data'][:, 0:i_end]
            radar.ngates = i_end






