import numpy as np
import matplotlib.pyplot as plt
import os
import time
import h5py
import yaml
from collision_predictor_mpc import COLPREDMPC_DATA_DIR
from collision_predictor_mpc import depth_inflate
import trimesh as tm


if __name__ == '__main__':
    dataset = 'half_clutter'  # cluttered or half_clutter or walls
    type = 'train'  # test or train
    outfile_name = 'half_clutter_train2.hdf5'

    print('loading data')
    with open(os.path.join(COLPREDMPC_DATA_DIR, dataset) + '.yaml','r') as f:
        metadata = yaml.full_load(f)
    depth_max = metadata['depth_max']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    shape_imgs = metadata['shape_imgs']
    width = shape_imgs[2]
    height = shape_imgs[1]
    nb_imgs = metadata['nb_imgs_'+type]
    h5file = h5py.File(os.path.join(COLPREDMPC_DATA_DIR, metadata['hdf5_train']), 'r')
    data = h5file['raw']

    h5_out = h5py.File(os.path.join(COLPREDMPC_DATA_DIR, outfile_name), 'w')
    h5_out.create_dataset('raw', (nb_imgs,1,270,480), dtype='float32')
    h5_out.create_dataset('collision', (nb_imgs,1,270,480), dtype='float32')

    drone_size = 0.25

    # mesh = tm.creation.box(extents=[drone_size, drone_size, drone_size])
    mesh = tm.creation.icosphere(radius=drone_size, subdivisions=2)

    tic = time.time()
    for img in range(nb_imgs):
        print('img ',img)
        depth_img = data[img,0,:,:]
        edges_inflated = depth_inflate.inflate_edges(depth_img, mesh, drone_size, depth_max, hfov)

        h5_out['raw'][img,0,:,:] = depth_img
        h5_out['collision'][img,0,:,:] = edges_inflated

        # print(time.time()-tic)
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(depth_img, vmin=0, vmax=1)
        # axs[1].imshow(edges_inflated, vmin=0, vmax=1)
        # plt.show()
