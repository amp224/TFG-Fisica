#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:05:45 2023

@author: amp
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import glorot_uniform
from tqdm.keras import TqdmCallback

from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer
from deepsphere import utils

from tqdm import tqdm

class auto_encoder(tf.keras.Model):
    def __init__(self, encoder_layer, decoder_layer, nside, indices):
        """
        Inits the auto-encoder with layers for the encoder and the decoder
        """
        # this line is necessary for all Model subclasses
        super(auto_encoder, self).__init__(name="")

        # save some properties
        self.nside = nside
        self.indices = indices

        # init the encoder and the decoder
        print("Initializing the encoder...")
        self.encoder = HealpyGCNN(nside=nside, indices=indices, layers=encoder_layer,
                                  n_neighbors=8)

        # get the bottle neck nside and indices
        self.bottle_neck_nside = self.encoder.nside_out
        self.bottle_neck_indices = self.encoder._transform_indices(nside_in=self.nside,
                                                        nside_out=self.bottle_neck_nside,
                                                        indices=self.indices)

        print("Initializing the decoder...")
        self.decoder = HealpyGCNN(nside=self.bottle_neck_nside,
                                  indices=self.bottle_neck_indices,
                                  layers=decoder_layer, n_neighbors=8)

    def summary(self, *args, **kwargs):
        """
        A wrapper for the summary routines of the decoder and encoder
        """
        print("Encoder summary:")
        self.encoder.summary(*args, **kwargs)
        print("Decoder summary:")
        self.decoder.summary(*args, **kwargs)

    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the autoencoder
        """
        bottle_neck = self.encoder(input_tensor, training=training, *args, **kwargs)
        reconstruction = self.decoder(bottle_neck, training=training, *args, **kwargs)

        return reconstruction


def minMaxRescale(x,a=0, b=1, diagnostic=False):
    """
    Performs  a MinMax Rescaling on an array 'x' to a generic range :math:`[a,b]`.
    """
    xmax = x.max()
    xmin = x.min()
    xresc = (b-a)*(x- xmin )/(xmax - xmin ) +a

    return (xresc,xmax,xmin) if diagnostic else xresc


alpha_LR = 0.1 # learning rate de LeakyReLU
K=5 # orden del polinomio de Chebyshev usado para aproximar




def dip_workflow(x_true,
                 f,
                 nside,
                 z_std=0.1,
                 loss_mask=None,
                 num_iters=5000,
                 init_lr=0.01):
    """Deep Image prior workflow
    Args:
        * x_true: Ground-truth image, only used for metrics comparison
        * f: Neural network to use as a prior
        * nside: 
        * loss_mask: if not None, a binary mask with the same shape as x0,
            which is applied to both x and x0 before applying the loss.
            Used for instance in the inpainting task.
        * num_iters: Number of training iterations
        * init_lr: Initial learning rate for Adam optimizer
    """
    # Sample input z
    shape = (1, 12 * nside ** 2,)
    
    z = tf.constant(np.random.uniform(low=0, high=1./10,
                                      size=shape).astype(np.float32) *\
                                      z_std, name='net_input')

    # reduce input images to [0,1]
    valmin = np.min(x_true)
    valmax = np.max(x_true)
    x_true = minMaxRescale(x_true, a=0, b=1)
    
    # Training Loss
    def loss_fn(x_true, x):
        if loss_mask is None:
            return tf.keras.losses.MSE(x, x_true)

        return tf.norm((x_true - x) * loss_mask) / tf.norm(x_true * loss_mask)


    def mse_to_gt(x_true, x):
        return tf.norm(x * loss_mask) / tf.norm(x_true * loss_mask)
        #return tf.norm(x) / tf.norm(x_true)
    
    # Optimization
    opt = tf.keras.optimizers.Adam(learning_rate=init_lr)
    f.compile(optimizer=opt, loss=loss_fn, metrics=mse_to_gt)
        
    
    history = f.fit(x=z, 
                    y=x_true, 
                    epochs=num_iters,
                    steps_per_epoch=1, 
                    verbose=0
                   )
    
    # Display results with gridspec
    x = f.predict(z)[0]
    # return x to physical values
    x = minMaxRescale(x, a=valmin, b=valmax)
    x_true = minMaxRescale(x_true, a=valmin, b=valmax)
    
    return x


encoder_layers = [hp_layer.HealpyPseudoConv(p=1, Fout=4, activation="elu"),
                  hp_layer.HealpyPseudoConv(p=1, Fout=8, activation="elu"),
                  hp_layer.HealpyPseudoConv(p=1, Fout=16, activation="elu"),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="elu"),
                  tf.keras.layers.LayerNormalization(axis=1),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="elu"),
                  tf.keras.layers.LayerNormalization(axis=1),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="linear"),]

decoder_layers = [hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="elu"),
                  tf.keras.layers.LayerNormalization(axis=1),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                           activation="elu"),
                  tf.keras.layers.LayerNormalization(axis=1),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                           activation="elu"),
                  tf.keras.layers.LayerNormalization(axis=1),
                  hp_layer.HealpyPseudoConv_Transpose(p=1, Fout=16, activation="elu"),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="elu"),
                  hp_layer.HealpyPseudoConv_Transpose(p=1, Fout=16, activation="elu"),
                  hp_layer.HealpyChebyshev(K=K, Fout=16, use_bias=True, use_bn=False, 
                                   activation="elu"),
                  hp_layer.HealpyPseudoConv_Transpose(p=1, Fout=1, activation="linear")]




nside = 128
npix = hp.nside2npix(nside)
indices = np.arange(npix)


model = auto_encoder(encoder_layers, decoder_layers, nside, indices)
model.build(input_shape=(None, len(indices), 1))

model.summary()

ind = 0
n_maps = 1

maps = np.load("Mapas/1000_prediction_maps_nested.npy")[n_maps*ind:n_maps*(ind+1),:,:]
mask = np.load('mask_downgraded_128_nested.npy')

mask[0,:,0] = hp.ud_grade(mask[0,:,0],nside)


start = time.time()

for i in range(n_maps):
    keras.backend.clear_session()
    model = auto_encoder(encoder_layers, decoder_layers, nside, indices)
    model.build(input_shape=(None, len(indices), 1))
    x_true = maps[i:(i+1),:,:]
    #x_true[0,:,0] = hp.ud_grade(x_true[0,:,0], nside)
    

    pred = dip_workflow(x_true, model, nside, num_iters=550,
                 loss_mask=mask)
    reconstructed = x_true*mask + pred*np.logical_not(mask)
    hp.mollview(pred[:,0], nest=True)
    hp.mollview(reconstructed[0,:,0], nest=True)
    hp.mollview(pred[:,0]-reconstructed[0,:,0], nest=True)
    hp.mollview(x_true[0,:,0], nest=True)
    np.save('dp_sin_combinar', pred)
    # maps[i,:,:] = x_true * np.int_(mask) + pred * np.logical_not(mask)

    

# np.save(f'output/inpainted/DP/{n_maps}_maps_inpainted_dp_set_{ind}', maps)
# np.save('mapa_0_dp', maps)

print(f'Tiempo requerido: {time.time()-start}s')

    
