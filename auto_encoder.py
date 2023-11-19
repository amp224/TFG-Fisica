#!/usr/bin/env python
# coding: utf-8

# # Generative Models
# 
# This notebook gives a quick example on how to use the transpose pseudo convolutions in DeepSphere to train a simple auto-encoder.
# 
# ![image.png](attachment:image.png)
# 
# The transpose pseudo convolutions use the healpy pixelization scheme to increase the nside of the data, making them useful for generative models.



#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from deepsphere import HealpyGCNN
from deepsphere import healpy_layers as hp_layer
from deepsphere import utils

from tqdm import tqdm
maps



training_data = np.load('Mapas/8000_training_maps_nested.npy')

    







class generator(tf.keras.Model):
    def __init__(self, encoder_layer, decoder_layer, nside, indices):
        """
        Inits the auto-encoder with layers for the encoder and the decoder
        """
        # this line is necessary for all Model subclasses
        super(generator, self).__init__(name="")
        
        # save some properties
        self.nside = nside
        self.indices = indices
        
        # init the encoder and the decoder
        print("Initializing the encoder...")
        self.encoder = HealpyGCNN(nside=nside, indices=indices, layers=encoder_layer,
                                  n_neighbors=20)
        
        # get the bottle neck nside and indices
        self.bottle_neck_nside = self.encoder.nside_out
        self.bottle_neck_indices = self.encoder._transform_indices(nside_in=self.nside, 
                                                        nside_out=self.bottle_neck_nside, 
                                                        indices=self.indices)
        
        print("Initializing the decoder...")
        self.decoder = HealpyGCNN(nside=self.bottle_neck_nside, 
                                  indices=self.bottle_neck_indices, 
                                  layers=decoder_layer, n_neighbors=20)
        
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


class discriminator(tf.keras.Model):
    def __init__(self, layer, nside, indices):
        """
        Inits the discriminator
        """
        # this line is necessary for all Model subclasses
        super(generator, self).__init__(name="")
        
        # save some properties
        self.nside = nside
        self.indices = indices
        
        # init the encoder and the decoder
        print("Initializing the encoder...")
        self.discriminator = HealpyGCNN(nside=nside, indices=indices, layers=encoder_layer,
                                  n_neighbors=20)
        
    def summary(self, *args, **kwargs):
        """
        A wrapper for the summary routines of the decoder and encoder
        """
        print("Encoder summary:")
        self.discriminator.summary(*args, **kwargs)
        
    def call(self, input_tensor, training=False, *args, **kwargs):
        """
        Calls the discriminator
        """
        validity = self.encoder(input_tensor, training=training, *args, **kwargs)
        
        return validity
    



K = 5
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
n_pix = hp.nside2npix(nside)
indices = np.arange(n_pix)
model = generator(encoder_layers, decoder_layers, nside, indices)
model.build(input_shape=(None, len(indices), 1))





mask = np.load('mask_downgraded_128_nested.npy')




def loss_fn(x_true, x):
    return tf.norm((x_true-x)*np.logical_not(mask)) / tf.norm(x_true*np.logical_not(mask))

def mse_to_gt(x_true, x):
    return tf.norm(x) / tf.norm(x_true)


model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss=loss_fn, 
              metrics=mse_to_gt)




print("Fit model on training data")
history = model.fit(
    x=training_data,
    y=training_data,
    batch_size=128,
    epochs=50,
)

del training_data

maps = np.load('Mapas/1000_prediction_maps_nested.npy')
reconstructed = model.predict(maps)
inpainted = maps * mask + reconstructed * np.logical_not(mask)

np.save('output/inpainted/AutoEncoder/1000_mapas_ae_inpainted', inpainted)



np.save('output/losses/loss_50_epochs', history.history['loss'])
np.save('output/losses/mse_to_gt_50_epochs', history.history['mse_to_gt'])

