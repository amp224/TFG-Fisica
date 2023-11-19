import numpy as np

maps = np.empty(shape=(1000,12*128**2,1))

for i in range(0,20):
    maps[50*i:50*(i+1),:,:] = np.load(f'output/inpainted/Diffusive/50_diff_inpainted_maps_128_set_{i}.npy')

np.save('output/inpainted/Diffusive/1000_mapas_diff_inpainted_128_ring', maps)


for i in range(0,20):
    maps[50*i:50*(i+1),:,:] = np.load(f'output/inpainted/DP/50_maps_inpainted_dp_set_{i}.npy')

np.save('output/inpainted/DP/1000_mapas_dp_inpainted_nested', maps)
