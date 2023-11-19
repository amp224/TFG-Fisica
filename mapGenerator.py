#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:40:08 2023

@author: amp
"""

import healpy as hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dl =  pd.read_csv('Mapas/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',
                        delim_whitespace=True, index_col = 0).L

# input_dl.Dl.plot(logx=False, logy=False, grid=True)
# plt.ylabel("$\dfrac{\ell(\ell+1)}{2\pi} C_\ell~[\mu K^2]$")
# plt.xlabel("$\ell$")
# plt.xlim([0, 128*3]);

lmax = input_dl.index[-1]

cl = input_dl.divide(input_dl.index * (input_dl.index + 1) / (2 * np.pi), 
                     axis='index')

cl  /= 1e12
cl = cl.reindex(np.arange(0, lmax+1))

cl = cl.fillna(0)

nside = 128

fwhm = 2.4 * np.sqrt(4*np.pi/hp.nside2npix(nside))

# número de mapas a generar
n_training = 8000
n_test = 2000
n_predict = 1000

sim_maps = np.empty(shape=(n_training, hp.nside2npix(nside), 1), 
                    dtype=np.float64)

for i in range(n_training):        
    mapa = hp.synfast(cl, nside=nside, pixwin=True, fwhm=fwhm)
    mapa = hp.reorder(mapa, r2n=True)
    sim_maps[i,:,0] = mapa

np.save(f'Mapas/{n_training}_training_maps_{nside}_nested', sim_maps)

sim_maps = np.empty(shape=(n_test, hp.nside2npix(nside), 1), 
                    dtype=np.float64)

for i in range(n_test):
    mapa = hp.synfast(cl, nside=nside, pixwin=True, fwhm=fwhm)
    mapa = hp.reorder(mapa, r2n=True)
    sim_maps[i,:,0] = mapa

np.save(f'Mapas/{n_test}_testing_maps_{nside}_nested', sim_maps)


#sim_maps = np.empty(shape=(n_predict, hp.nside2npix(nside), 1), 
#                    dtype=np.float64)
#for i in range(n_predict):        
#    mapa = hp.synfast(cl, nside=nside, pixwin=True, fwhm=fwhm)
#    sim_maps[i,:,0] = mapa

#np.save(f'Mapas/{n_predict}_prediction_maps_{nside}_ring', sim_maps)

#for i in range(n_predict):
#    mapa = hp.reorder(sim_maps[i,:,0], r2n=True)
#    sim_maps[i,:,0] = mapa

#np.save(f'Mapas/{n_predict}_prediction_maps_{nside}_nested', sim_maps)
