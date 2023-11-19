import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import healpy as hp
import time





class DiffusiveInpainting:
    """
    Attributes:
        nside: Nside del mapa
        mapa: Numpy array unidimensional que representa el mapa en formato RING de healpy
        mask: Máscara del mapa (0 para los píxeles enmascarados)
        masked_pix: Índices de los píxeles enmascarados
    """
    
    def __init__(self, mapa, mask):
        self.nside = hp.get_nside(mapa)
        self.mapa = mapa
        self.mask = mask
        self.masked_pix = [i for i in range(len(self.mask)) if self.mask[i]==0]

        
    def non_zero_neighbours(self):
        """
        Devuelve un array con el número de vecinos enmascarados de cada píxel también enmascarado
        """
        neighbours = np.empty(8)
        n_nz = np.empty(len(self.masked_pix))
        
        for i, pix in enumerate(self.masked_pix):
            neighbours = hp.pixelfunc.get_all_neighbours(self.nside, pix)
            n_nz[i] = len([_ for _ in neighbours if _ == 0])
        
        return n_nz
        
    
    def sort_by_neighbours(self):
        """
        Devuelve los índices de los píxeles enmascarados, ordenados de menor a mayor número
        de vecinos enmascarados
        """
        # obtiene los índices de los pixeles enmascarados
        # ordenados de menor a mayor número de vecinos no nulos
        ind = self.non_zero_neighbours().argsort()
        # queremos primero los píxeles con más vecinos buenos
        ind = ind[::-1]
        # ordenamos los píxeles según los índices obtenidos
        return np.array(self.masked_pix)[ind.astype(int)]

    
    def inp_pixel(self, m, pix):
        """
        Interpola el valor de un pixel con índice dado tomando el valor medio de sus vecinos
        """
        neighbours = hp.pixelfunc.get_all_neighbours(self.nside, pix)
        return np.mean([m[p] for p in neighbours if p != -1])

    
    def diff_inpaint(self, epsi=1e-3):
        """
        Usa el método de Jacobi para hacer inpainting sobre el mapa enmascarado.
        """
        npix = hp.nside2npix(self.nside)
        #orden = self.sort_by_neighbours()
        mapa = self.mapa
        mapa_inp = np.multiply(mapa,self.mask)
        parada = np.inf
        historial = []
        
        while parada > epsi:
            for i in self.masked_pix:
                mapa_inp[i] = self.inp_pixel(mapa, i)
            parada = 1/len(mapa[mapa!=0]) * np.sum([abs((mapa_inp[i]-mapa[i])/mapa[i]) for i in self.masked_pix if mapa[i] != 0])
            historial.append(parada)
            mapa = np.copy(mapa_inp)
        return mapa_inp, historial


ind = 19  
n_mapas = 50

maps = np.load('Mapas/1000_prediction_maps_128_ring.npy')
maps = maps[(n_mapas*ind):(n_mapas*(ind+1)),:,]
mask = (np.load('Mapas/mask_downgraded_128_ring.npy'))[0,:,0]
nside = hp.get_nside(maps[0,:,0])
npix = hp.nside2npix(nside)


inpainted_maps = np.empty(shape=(n_mapas,npix,1)) # inpainting done only in Temperature -> only one channel

history = []

start = time.time()
for i in range(0, n_mapas):
    obj = DiffusiveInpainting(maps[i,:,0], mask)
    inpainted_maps[i,:,0], _ = obj.diff_inpaint()
print(f'Tiempo: {time.time()-start}')



np.save(f"output/inpainted/Diffusive/{n_mapas}_diff_inpainted_maps_{nside}_set_{ind}",
        inpainted_maps)
