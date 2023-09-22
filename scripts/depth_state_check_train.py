import numpy as np
import torch
from collision_predictor_mpc.colpred.depth_state_check import Colpred
from collision_predictor_mpc.colpred.collision_checker import ColChecker
from collision_predictor_mpc.colpred.losses import loss_weighted_bce, loss_KLD, loss_spatial_gradient, loss_classif_relus
from collision_predictor_mpc import COLPREDMPC_DATA_DIR, COLPREDMPC_LOG_DIR
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics
import pickle as pkl
import h5py
import yaml
import matplotlib.pyplot as plt
from torchinfo import summary
import time
import os
import torchvision


class DataCollisionCheck(Dataset):
    """Dataset for classifier training. The augmentation is performed on the numpy images, since the warp kernel (colcheck) requires input array to be on cpu."""
    def __init__(self, imgs, collision_imgs, idx, augment=False, seed=None):
        self.imgs = imgs
        self.collision_imgs = collision_imgs
        self.idx = idx
        self.augment = augment
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.std_depth = 0.01  # std for white Gaussian noising image and depth (meters normalized by depthmax)
        #self.shift = 0.1  # maximum absolute shifting of depth images
        self.gpu = 'cuda'


    def __len__(self):
        return len(self.idx)


    def __getitem__(self, idx):
        img_gpu = torch.tensor(self.imgs[self.idx[idx],:,:,:], device=self.gpu)
        col_gpu = torch.tensor(self.collision_imgs[self.idx[idx],:,:,:], device=self.gpu)
        ## data augmentation:
        if self.augment:
            # shift = self.rng.uniform(-self.shift, self.shift)
            # img_gpu = (img_gpu + shift).clamp(0,1)
            # col_gpu = (col_gpu + shift).clamp(0,1)
            if self.rng.uniform() < 0.5:
                img_gpu = torchvision.transforms.RandomHorizontalFlip(p=1)(img_gpu)
                col_gpu = torchvision.transforms.RandomHorizontalFlip(p=1)(col_gpu)
            if self.rng.uniform() < 0.5:
                img_gpu = torchvision.transforms.RandomVerticalFlip(p=1)(img_gpu)
                col_gpu = torchvision.transforms.RandomVerticalFlip(p=1)(col_gpu)

        col_cpu = col_gpu.detach().clone().cpu()

        if self.augment:
            img_gpu = (img_gpu + torch.randn_like(img_gpu, device=self.gpu)*self.std_depth).clamp(0,1)

        return img_gpu, col_cpu



def samples_points(colcheck, imgs, nb_frustrum, nb_ball, device):
    """Sample Cartesian points, and checks the corresponding image for generating collision label. Returns the normalized state and label tensors."""
    ## sample states and get collision labels
    states_frustrum, states_frustrum_norm = colcheck.sample_pos_in_frustrum(imgs.shape[0]*nb_frustrum, add_margin=True)
    states_ball, states_ball_norm = colcheck.sample_pos_in_safeball(imgs.shape[0]*nb_ball, add_margin=True)
    ## concatenate both state arrays such that each block of points_per_img rows contains nb_frustrum and nb_ball points in the respective ranges
    states = np.hstack([states_frustrum.reshape((-1,nb_frustrum,3)), states_ball.reshape((-1,nb_ball,3))]).reshape((-1,3))
    states_norm = np.hstack([states_frustrum_norm.reshape((-1,nb_frustrum,3)), states_ball_norm.reshape((-1,nb_ball,3))]).reshape((-1,3))

    labels = colcheck.check_image_points(imgs[:,0,:,:], states)
    labels = torch.tensor(labels, device=device)
    states = torch.tensor(states, device=device)
    states_norm = torch.tensor(states_norm, device=device, requires_grad=True)  # requires_grad=True is required to compute the gradients of output vs inputs in cost

    return states, states_norm, labels



if __name__ == '__main__':
    ## parameters
    dataset = 'half_clutter'  # cluttered or half_clutter or walls
    size_latent = 128
    train_valid_ratio = 0.9  # ratio of train and valid data
    learning_rate = 1e-6
    nb_epochs = 100
    batch_eval_size = 100  # nb of batches for evaluation
    batch_img_size = 500  # nb of images batches together
    points_per_img = 20000  # nb of points sampled per image
    subbatch_size = 1000000  # actual size of batches (state, img) processed together during training
    nb_subbatch = int(points_per_img * batch_img_size / subbatch_size)
    nb_img_subbatch = int(subbatch_size / points_per_img)
    assert (points_per_img * batch_img_size / subbatch_size) % 1 == 0
    assert (subbatch_size / points_per_img) % 1 == 0
    dmax_grad = 0.25  # minimum distance from obstacles at which colpred starts to increase
    ratio_points_ball = 0.15  # ratio of points that are sampled of the ball around origin rather than in frustrum
    assert (ratio_points_ball*points_per_img)%1 == 0
    nb_frustrum = int(points_per_img*(1-ratio_points_ball))
    nb_ball = int(points_per_img*ratio_points_ball)
    safe_ball_size = 0.1  # size of safe ball around camera within which collision label is 0 [m]
    weights_bce = [1,25]  # weights on 0 and 1 classes in BCE

    ## data
    print('loading data')
    with open(os.path.join(COLPREDMPC_DATA_DIR, dataset) + '.yaml','r') as f:
        metadata = yaml.full_load(f)
    depth_max = metadata['depth_max']
    hfov = metadata['hfov']
    vfov = metadata['vfov']
    shape_imgs = metadata['shape_imgs']
    nb_imgs = metadata['nb_imgs_train']
    h5file = h5py.File(os.path.join(COLPREDMPC_DATA_DIR, metadata['hdf5_train']), 'r')
    data_input = h5file['raw']
    data_collision = h5file['collision']
    idx_data_train = range(int(nb_imgs*train_valid_ratio))
    idx_data_valid = range(int(nb_imgs*train_valid_ratio), nb_imgs)

    ## get nn
    # colpred = Colpred(size_latent=size_latent, dropout_rate=0.2, filename=COLPREDMPC_LOG_DIR+'/inflated_retry')
    colpred = Colpred(size_latent=size_latent, dropout_rate=0.2, filename=COLPREDMPC_LOG_DIR+'/inflated')
    colpred.load_weights()
    # colpred.linear.load_weights()
    # colpred.encoder.load_weights()
    # colpred.encoder.eval()
    # colpred.encoder.requires_grad_(False)

    print('using', colpred.device)
    summary(colpred, input_size=[(1, 3), (1, shape_imgs[0], shape_imgs[1], shape_imgs[2])])

    ## get collision checker
    colcheck = ColChecker(depth_max, hfov, vfov, safe_ball_size)

    ## dataloaders
    train_dataloader = DataLoader(DataCollisionCheck(data_input, data_collision, idx_data_train, augment=True), batch_size=batch_img_size, shuffle=True)
    valid_dataloader = DataLoader(DataCollisionCheck(data_input, data_collision, idx_data_valid, augment=False), batch_size=batch_eval_size, shuffle=False)

    nb_train_batches = len(train_dataloader)
    nb_valid_batches = len(valid_dataloader)


    ## training
    loss_train = np.zeros((nb_epochs, nb_train_batches, 3))  # store both losses (bce, mse) across batches for trainign
    loss_valid = np.zeros((nb_epochs, nb_valid_batches, 3))  # store both losses (bce, mse) across batches for validation
    metrics = np.zeros((nb_epochs, nb_valid_batches, 3))  # store validation metrics (accuracy, precision, recall) acress batches
    fig = plt.figure()
    optimizer = torch.optim.Adam(colpred.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(colpred.linear.parameters(), lr=learning_rate)
    for idx_epoch in range(nb_epochs):
        tic = time.time()
        print(f'-------------------------------\nepoch {idx_epoch+1}')
        ## train
        colpred.train()
        # colpred.linear.train()
        for idx_batch, (imgs_gpu, cols_cpu) in enumerate(train_dataloader):
            optimizer.zero_grad()
            states, states_norm, labels = samples_points(colcheck, cols_cpu, nb_frustrum, nb_ball, colpred.device)

            idxs_sample_state = np.arange(states_norm.shape[0]).reshape((-1,subbatch_size))
            idxs_sample_img = np.arange(imgs_gpu.shape[0]).reshape((-1,nb_img_subbatch))
            for idx_subbatch in range(idxs_sample_state.shape[0]):
                range_sample_state = idxs_sample_state[idx_subbatch,:]
                range_sample_img = idxs_sample_img[idx_subbatch,:]
                in_imgs = imgs_gpu[range_sample_img,:]
                check_img_col = cols_cpu[range_sample_img,:]
                in_states_norm = states_norm[range_sample_state,:]
                in_states_unorm = states[range_sample_state,:]
                out_labels = labels[range_sample_state]

                nn_mean, nn_logvar = colpred.encoder(in_imgs)
                std = torch.exp(0.5 * nn_logvar)
                eps = torch.randn_like(std)
                latents = eps * std + nn_mean
                latents = latents.repeat_interleave(points_per_img, 0)

                nn_classif = colpred.linear(in_states_norm, latents)
                loss_classif = 10*loss_weighted_bce(nn_classif.flatten(), out_labels.flatten(), weights_bce)
                loss_kld = loss_KLD(nn_mean, nn_logvar, 1., size_latent, shape_imgs[1:])
                # loss_gradient = 0.2*loss_spatial_gradient(nn_classif.flatten(), in_states_unorm, in_states_norm, dmax_grad, safe_ball_size, depth_max)
                loss_gradient = 0*loss_kld
                loss_train[idx_epoch, idx_batch, 0] += loss_classif.item()
                loss_train[idx_epoch, idx_batch, 1] += loss_kld.item()
                loss_train[idx_epoch, idx_batch, 2] += loss_gradient.item()

                ## backpropagation
                loss = loss_classif + loss_kld + loss_gradient
                loss.backward()
                optimizer.step()
            loss_train[idx_epoch,idx_batch,:] /= nb_subbatch

            ## print
            # if (idx_batch+1) % (nb_train_batches/5) == 0:
            print(f'train batch: {idx_batch+1}/{nb_train_batches} - loss (classif/kld/grad): {loss_train[idx_epoch,idx_batch,0]:>8f} / {loss_train[idx_epoch,idx_batch,1]:>8f} / {loss_train[idx_epoch,idx_batch,2]:>8f}')
            print(f'elapsed time: {time.time()-tic}')

            # colpred.linear.save_weights()
            colpred.save_weights_idx(idx_epoch+1)
            # colpred.save_weights()

        ## valid
        colpred.eval()
        for idx_batch, (imgs_gpu, cols_cpu) in enumerate(valid_dataloader):
            in_states_unorm, in_states_norm, out_labels = samples_points(colcheck, cols_cpu, nb_frustrum, nb_ball, colpred.device)

            ## forward & loss
            nn_mean, nn_logvar = colpred.encoder(imgs_gpu)
            std = torch.exp(0.5 * nn_logvar)
            eps = torch.randn_like(std)
            latents = eps * std + nn_mean
            latents = latents.repeat_interleave(points_per_img, 0)

            nn_classif = colpred.linear(in_states_norm, latents)
            loss_classif = 10*loss_weighted_bce(nn_classif.flatten(), out_labels.flatten(), weights_bce)
            loss_kld = loss_KLD(nn_mean, nn_logvar, 1., size_latent, shape_imgs[1:])
            # loss_gradient = 0.2*loss_spatial_gradient(nn_classif.flatten(), in_states_unorm, in_states_norm, dmax_grad, safe_ball_size, depth_max)
            loss_gradient = 0*loss_kld
            loss_valid[idx_epoch, idx_batch, 0] += loss_classif.item()
            loss_valid[idx_epoch, idx_batch, 1] += loss_kld.item()
            loss_valid[idx_epoch, idx_batch, 2] += loss_gradient.item()

            if (idx_batch+1) % (nb_valid_batches/2) == 0:
                print(f'valid batch: {idx_batch+1}/{nb_valid_batches} - loss (classif/kld/grad): {loss_classif.item():>8f} / {loss_kld.item():>8f} / {loss_gradient.item():>8f}')
                print(f'elapsed time: {time.time()-tic}')

            ## metrics
            target = out_labels.to(torch.bool).detach().cpu().numpy()
            pred = torch.where(nn_classif.flatten() > 0.5, 1, 0).to(torch.bool).detach().cpu().numpy()
            metrics[idx_epoch, idx_batch, 0] += sklearn.metrics.accuracy_score(target, pred)
            metrics[idx_epoch, idx_batch, 1] += sklearn.metrics.precision_score(target, pred)
            metrics[idx_epoch, idx_batch, 2] += sklearn.metrics.recall_score(target, pred)

        toc = time.time()

        print(f'(mean,std) train error (classif/kld/grad): {loss_train[idx_epoch,:,0].mean():>8f} +- {loss_train[idx_epoch,:,0].std():>8f} / {loss_train[idx_epoch,:,1].mean():>8f} +- {loss_train[idx_epoch,:,1].std():>8f} / {loss_train[idx_epoch,:,2].mean():>8f} +- {loss_train[idx_epoch,:,2].std():>8f}')
        print(f'(mean,std) valid error (classif/kld/grad): {loss_valid[idx_epoch,:,0].mean():>8f} +- {loss_valid[idx_epoch,:,0].std():>8f} / {loss_valid[idx_epoch,:,1].mean():>8f} +- {loss_valid[idx_epoch,:,1].std():>8f} / {loss_valid[idx_epoch,:,2].mean():>8f} +- {loss_valid[idx_epoch,:,2].std():>8f}')
        print(f'(mean,std) valid accuracy: {metrics[idx_epoch,:,0].mean():>8f} +- {metrics[idx_epoch,:,0].std():>8f}')
        print(f'(mean,std) valid precision: {metrics[idx_epoch,:,1].mean():>8f} +- {metrics[idx_epoch,:,1].std():>8f}')
        print(f'(mean,std) valid recall: {metrics[idx_epoch,:,2].mean():>8f} +- {metrics[idx_epoch,:,2].std():>8f}')
        print(f'epoch time: {toc-tic}')

        ## save
        # colpred.save_weights_idx(idx_epoch+1)
        # colpred.save_weights()
        # colpred.linear.save_weights()
        # colpred.encoder.save_weights()

        fig.clear()
        plt.plot(loss_train[:idx_epoch+1,:,:].sum(axis=2).mean(axis=1), label='training loss')
        plt.plot(loss_valid[:idx_epoch+1,:,:].sum(axis=2).mean(axis=1), label='validation loss')
        plt.legend()
        plt.grid()
        fig.savefig(colpred.filename + '.png')

    # colpred.save_weights()
    # colpred.encoder.save_weights()
    # colpred.linear.save_weights()

    print('done!')
