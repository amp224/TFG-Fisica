#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:33:22 2023

@author: amp
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import deepsphere
from tqdm import tqdm

def spectrum(maps1, maps2=None, filename=None,
               dl=False, summ=False, show=False):

    
    
    n_maps = np.shape(maps1)[0]
    lmax = 3*hp.get_nside(maps1[0,:,0]) - 1
    ll = np.arange(lmax + 1)
    spectrums = np.empty((n_maps, lmax+1))
    
    if maps2 is not None:
        assert np.shape(maps1) == np.shape(maps2)
        for i in tqdm(range(n_maps)):
            spectrums[i,:] = hp.anafast(maps1[i,:,0], maps2[i,:,0], lmax=lmax)
    else:     
        for i in tqdm(range(n_maps)):
            spectrums[i,:] = hp.anafast(maps1[i,:,0], lmax=lmax, 
                                        use_pixel_weights=True) 

    
    if dl:
        spectrums = ll*(ll+1)/2/np.pi * spectrums
        
    
    spec_avg = np.mean(spectrums, axis=0)
    spec_std_dev = np.std(spectrums, axis=0)
    
    
    if show:
        if n_maps>1:
            plt.fill_between(ll, spec_avg-spec_std_dev, 
                             spec_avg+spec_std_dev, alpha=0.5)
        plt.xscale('symlog',linthresh=20)
        plt.xlim(left=0, right=lmax)
        plt.xlabel('$\ell$')
        plt.ylabel('$D_\ell / \mu K^2$')
        plt.plot(ll, spec_avg)
        plt.show()
    
    if filename is not None:
            np.save(filename, spectrums)
        
    
    return (spectrums, spec_avg, spec_std_dev) if summ else spectrums

    


# def cross_spectrum(maps1, maps2, filename=None, dl=False, 
#                    summ=False, show=False):

#     assert np.shape(maps1) == np.shape(maps2)

#     n_maps = np.shape(maps1)
#     lmax = 3*hp.get_nside(maps1[0,:]) - 1
#     ll = np.arange(lmax+1)
#     cross_specs = np.empty((shape[0], lmax+1))

#     for i in np.arange(shape[0]):
#         cross_specs = hp.anafast(maps1[i,:,0], maps2[i,:,0], lmax=lmax)

#     cross_avg = np.mean(cross_specs, axis=0)
#     cross_std_dev = np.std(cross_specs, axis=0)

#     if filename is not None:
#         np.save(filename, cross_specs)

#     if show:
#         if shape[0] > 1:
#             plt.fill_between(ll, cross_avg-cross_std_dev,
#                              cross_avg+cross_std_dev, alpha)
#         plt.xscale('symlog', linthresh=100)
#         plt.plot(ll, cross_avg)
#         plt.show()


# def statistics(maps, method):

    





