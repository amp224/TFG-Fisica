#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:19:10 2023

@author: amp
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
from tqdm import tqdm

diff = np.load('inpainted/Diffusive/mean_diff_pow_spec_dl.npy')
sd_diff = np.load('inpainted/Diffusive/sd_diff_pow_spec_dl.npy')
corr_diff = np.load('inpainted/Diffusive/corr_matrix_diff_cl.npy')

dp = np.load('inpainted/DeepPrior/mean_dp_pow_spec_dl.npy')
sd_dp = np.load('inpainted/DeepPrior/sd_dp_pow_spec_dl.npy')
corr_dp = np.load('inpainted/DeepPrior/corr_matrix_dp_cl.npy')

true = np.load('Mapas/mean_ori_pow_spec_dl.npy')
sd_true = np.load('Mapas/sd_ori_pow_spec_dl.npy')
corr_true = np.load('Mapas/corr_matrix_cl.npy')

masked = np.load('inpainted/Masked/mean_masked_pow_spec_dl.npy')
sd_masked = np.load('inpainted/Masked/sd_masked_pow_spec_dl.npy')
corr_masked = np.load('inpainted/Masked/corr_matrix_masked_cl.npy')

nside = 128

lmax = 3*nside -1

ll = np.arange(lmax+1)


plt.figure(figsize=(9,6))
plt.fill_between(ll, diff-sd_diff, diff+sd_diff, alpha=0.2)
plt.plot(ll, diff, label='Diffusive')

plt.fill_between(ll, dp-sd_dp, dp+sd_dp, alpha=0.2)
plt.plot(ll, dp, label='Deep Prior')

plt.fill_between(ll, true-sd_true, true+sd_true, alpha=0.2, ec='orange')
plt.plot(ll, true, label='Model')

plt.legend()

# adjust x-axis aesthetic
plt.xlim([2,None])
plt.xscale('symlog', linthresh=30)
plt.xlabel('$\ell$')

x_ticks = [2,5,10,15,20,25,30,40,55,75,100,150,200,250,300]
tick_labels = [str(x) for x in x_ticks]
plt.xticks(x_ticks, tick_labels )

# adjust y-axis aesthetic
plt.ylabel('$D_\ell / \mu$K$^2$')
plt.ylim([0,3000])
plt.plot()


fig, axs = plt.subplots(2,2, figsize=(11,11), layout='constrained')

         

axs[0,0].imshow(corr_true, origin='lower')
axs[0,0].set_title('Model')
axs[0,0].set_xlabel('$\ell$')
axs[0,0].set_ylabel('$\ell$')

axs[0,1].imshow(corr_masked, origin='lower')
axs[0,1].set_title('Masked')
axs[0,1].set_xlabel('$\ell$')
axs[0,1].set_ylabel('$\ell$')

axs[1,0].imshow(corr_diff, origin='lower')
axs[1,0].set_title('Diffusive')
axs[1,0].set_xlabel('$\ell$')
axs[1,0].set_ylabel('$\ell$')

axs[1,1].imshow(corr_dp, origin='lower')
axs[1,1].set_title('Deep Prior')
axs[1,1].set_xlabel('$\ell$')
axs[1,1].set_ylabel('$\ell$')

fig.colorbar(axs[1,1].imshow(corr_dp, origin='lower'), ax=axs.ravel().tolist())
plt.show()

