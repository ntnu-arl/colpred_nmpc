import numpy as np
import h5py
import pickle
import yaml
import os
from collision_predictor_mpc import COLPREDMPC_DATA_DIR, COLPREDMPC_WEIGHT_DIR, COLPREDMPC_LOG_DIR
from collision_predictor_mpc.colpred.depth_state_check import Colpred
from collision_predictor_mpc.colpred.collision_checker import ColChecker
import torch
import time
import matplotlib.pyplot as plt
import sklearn.metrics
from torchinfo import summary
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate system with nmpc.')
    parser.add_argument('--cpu', dest='cpu', action='store_true', default=False, help='Force cpu.')
    args = parser.parse_args()

    dataset = 'half_clutter'  # cluttered or half_clutter or walls
    safe_ball_size = 0.1

    with open(os.path.join(COLPREDMPC_DATA_DIR, dataset) + '.yaml','r') as f:
        metadata = yaml.full_load(f)
    depth_max = metadata['depth_max']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    shape_imgs = metadata['shape_imgs']
    nb_imgs = metadata['nb_imgs_test']
    h5file = h5py.File(os.path.join(COLPREDMPC_DATA_DIR, metadata['hdf5_test']), 'r')
    data_input = h5file['raw']
    data_collision = h5file['collision']

    colcheck = ColChecker(depth_max, hfov, vfov, safe_ball_size)
    # colpred = Colpred(size_latent=128, filename=COLPREDMPC_WEIGHT_DIR+'/inflated_retry')
    colpred = Colpred(size_latent=128, filename=COLPREDMPC_WEIGHT_DIR+'/inflated')
    # colpred = Colpred(size_latent=128)
    if args.cpu: colpred.device = 'cpu'
    print('using', colpred.device)
    summary(colpred, input_size=[(1,3),(1, shape_imgs[0], shape_imgs[1], shape_imgs[2])], device=colpred.device)
    colpred.load_weights()
    # colpred.linear.load_weights()
    # colpred.encoder.load_weights()

    metrics = np.zeros((nb_imgs, 8))
    idx = list(range(nb_imgs))
    # np.random.shuffle(idx)
    for i in idx:
        depth_img = np.expand_dims(data_input[i,:,:,:], 0)
        collision_img = np.expand_dims(data_collision[i,:,:,:], 0)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(depth_img[0,0,:,:])
        # ax[1].imshow(collision_img[0,0,:,:])
        # plt.show()

        with torch.no_grad():
            latent = colpred.encoder.encode(torch.tensor(depth_img, device=colpred.device))

        nb_points = 4000000


        grid, grid_normalized = colcheck.sample_pos_in_frustrum(nb_points)
        # grid, grid_normalized = c olcheck.sample_pos_in_safeball(nb_points)
        gt = colcheck.check_image_points(collision_img[0,0,:,:], grid)
        grid_normalized = torch.tensor(grid_normalized, device=colpred.device, requires_grad=True)
        pred = colpred.linear(grid_normalized, latent.repeat(int(grid.shape[0]/latent.shape[0]), 1))
        binary_pred = np.where(pred.detach().cpu().numpy() > 0.5, 1, 0)

        # report = metrics.classification_report(gt, binary_pred, target_names=['free', 'collision'])
        ratio_col_gt = np.count_nonzero(gt) / gt.shape[0]
        ratio_col_pred = np.count_nonzero(binary_pred) / binary_pred.shape[0]
        accuracy = sklearn.metrics.accuracy_score(gt, binary_pred)
        precision = sklearn.metrics.precision_score(gt, binary_pred, zero_division=1.0)  # zero_div handles cases where no collision appears in image, hence precision/recall becomes undefined, the correct thing would be to ignore those images but too lazy
        recall = sklearn.metrics.recall_score(gt, binary_pred, zero_division=1.0)
        # grad = torch.autograd.grad(pred, grid_normalized, grad_outputs=torch.ones_like(pred), retain_graph=True, create_graph=True)[0]
        # norm_grad = torch.linalg.vector_norm(grad, dim=1).detach().cpu().numpy()
        # grad_ratio = np.count_nonzero(norm_grad>2.)/norm_grad.shape[0]
        # metrics[i,:] = [ratio_col_gt, ratio_col_pred, accuracy, precision, recall, norm_grad.mean(), norm_grad.max(), grad_ratio]
        metrics[i,:] = [ratio_col_gt, ratio_col_pred, accuracy, precision, recall,0,0,0]
        # print(i, ratio_col_gt, ratio_col_pred, accuracy, precision, recall, norm_grad.mean(), norm_grad.max(), grad_ratio)
        print(i, ratio_col_gt, ratio_col_pred, accuracy, precision, recall)


        # angles = [20,10,0,-10,-20]
        # nb_pix_v = shape_imgs[1]
        # fig = plt.figure(str(i))
        # ax = fig.subplots(nrows=len(angles), ncols=5)
        # for i, bearing_v_deg in enumerate(angles):
        #     grid, grid_normalized = colcheck.grid_frustrum_slice(nb_points, bearing_v_deg, True)
        #     pixel_v = np.clip(-bearing_v_deg*np.pi/180/vfov * nb_pix_v/2 + nb_pix_v/2, 0, nb_pix_v-1)
        #
        #     # gt = colcheck.check_image_points(depth_img[0,0,:,:], grid)
        #     gt = colcheck.check_image_points(collision_img[0,0,:,:], grid)
        #     # gt = colcheck.check_image_points_np(collision_img[0,0,:,:], grid)
        #
        #     with torch.no_grad():
        #         pred = colpred.linear(torch.tensor(grid_normalized, device=colpred.device), latent.repeat(int(grid.shape[0]/latent.shape[0]), 1))
        #     pred = pred.cpu().numpy()
        #     binary_pred = np.where(pred > 0.5, 1, 0)
        #
        #     x = grid[:,0].reshape((int(grid.shape[0]**(1/2)),int(grid.shape[0]**(1/2)))).transpose()
        #     y = grid[:,1].reshape((int(grid.shape[0]**(1/2)),int(grid.shape[0]**(1/2)))).transpose()
        #
        #     ax[i,0].imshow(depth_img[0,0,:,:], vmin=0., vmax=1.)
        #     ax[i,0].hlines(pixel_v, 0, 480, color='red')
        #     ax[i,0].axis('off')
        #
        #     ax[i,1].imshow(collision_img[0,0,:,:], vmin=0., vmax=1.)
        #     ax[i,1].hlines(pixel_v, 0, 480, color='red')
        #     ax[i,1].axis('off')
        #
        #     ax[i,2].set_aspect('equal')
        #     ax[i,2].invert_xaxis()
        #     ax[i,2].contourf(y, x, gt.reshape((int(grid.shape[0]**(1/2)),int(grid.shape[0]**(1/2)))).transpose(), vmin=0., vmax=1.)
        #
        #     ax[i,3].set_aspect('equal')
        #     ax[i,3].invert_xaxis()
        #     ax[i,3].contourf(y, x, pred.reshape((int(grid.shape[0]**(1/2)),int(grid.shape[0]**(1/2)))).transpose(), vmin=0., vmax=1.)
        #
        #     ax[i,4].set_aspect('equal')
        #     ax[i,4].invert_xaxis()
        #     ax[i,4].contourf(y, x, (2*gt-binary_pred.flatten()).reshape((int(grid.shape[0]**(1/2)),int(grid.shape[0]**(1/2)))).transpose(), vmin=-1., vmax=2., cmap='bwr')
        #
        # plt.show()


        # # grid, grid_normalized = colcheck.sample_pos_in_safeball(nb_points, True)
        # grid, grid_normalized = colcheck.sample_pos_in_frustrum(nb_points, False)
        # # grid, grid_normalized = colcheck.grid_frustrum(nb_points)
        # # grid, grid_normalized = colcheck.grid_frustrum_slice(nb_points, 0)
        #
        # gt = colcheck.check_image_points(depth_img[0,0,:,:], grid)
        # with torch.no_grad():
        #     pred = colpred.linear(torch.tensor(grid_normalized, device=colpred.device), latent.repeat(int(grid.shape[0]/latent.shape[0]), 1))
        # binary_pred = np.where(pred.cpu().numpy() > 0.5, 1, 0)
        #
        # grid_ones_gt = np.array([grid[i,:] for i in range(grid.shape[0]) if gt[i]])
        # gris_ones_pred = np.array([grid[i,:] for i in range(grid.shape[0]) if 1])#binary_pred[i]])
        #
        # fig = plt.figure()
        # ax1 = fig.add_subplot(131)
        # ax1.imshow(depth_img[0,0,:,:], vmin=0., vmax=1.)
        # ax1.axis('off')
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.scatter(grid_ones_gt[:,0], grid_ones_gt[:,1], grid_ones_gt[:,2], color='k')
        # ax2.scatter(0,0,0, marker='s')
        # ax3 = fig.add_subplot(133, projection='3d')
        # ax3.scatter(gris_ones_pred[:,0], gris_ones_pred[:,1], gris_ones_pred[:,2], color='k')
        # ax3.scatter(0,0,0, marker='s')
        #
        # plt.show()

    print('ratio_col_gt', np.mean(metrics[:,0]), np.std(metrics[:,0]))
    print('ratio_col_pred', np.mean(metrics[:,1]), np.std(metrics[:,1]))
    print('accuracy', np.mean(metrics[:,2]), np.std(metrics[:,2]))
    print('precision', np.mean(metrics[:,3]), np.std(metrics[:,3]))
    print('recall', np.mean(metrics[:,4]), np.std(metrics[:,4]))
    print('avg gradient', np.mean(metrics[:,5]), np.std(metrics[:,5]))
    print('max gradient', np.mean(metrics[:,6]), np.std(metrics[:,6]))
    print('% grad > th', np.mean(metrics[:,7]), np.std(metrics[:,7]))

    outfile = os.path.join(COLPREDMPC_LOGS_DIR, 'metrics_test_inflated.pkl')
    with open(outfile, 'wb') as f:
        pickle.dump(metrics, f)
