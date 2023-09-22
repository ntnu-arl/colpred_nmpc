import numpy as np
import torch


def loss_MSE_valid_pixels(y, x):
    """MSE loss for VAE ignoring invalid pixels."""
    mse = torch.nn.functional.mse_loss(y, x, reduction="none")
    mse = torch.where(y >= 0, mse, 0)
    mse = torch.mean(torch.sum(mse, dim=[1, 2, 3]))
    return mse


def loss_KLD(mean, logvar, beta=1., size_latent=128, size_img=(270, 480)):
    """KLD loss for VAE with normalized beta parameter, see https://openreview.net/pdf?id=Sy2fzU9gl."""
    beta_norm = (beta*size_latent)/(size_img[0]*size_img[1])
    kld_term = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
    kld = kld_term * beta_norm
    return kld


def loss_weighted_bce(predictions, labels, weights=[1,1]):
    """BCE classification loss with different weighting on both classes.
    weights has of 2 elements: w[0] and w[1] are respectively the weights on 0 and 1 classes.
    """
    clamped = torch.clamp(predictions, min=1e-7, max=1-1e-7)  # avoiding numerical instabilities
    bce = - weights[1] * labels * torch.log(clamped) - weights[0] * (1-labels) * torch.log(1 - clamped)
    return torch.mean(bce)


def loss_classif_relus(predictions, labels):
    zeros = torch.where(labels==0., predictions-0.5, 0)
    ones = torch.where(labels==1., 0.5-predictions, 0)
    relus_zeros = torch.nn.functional.relu(zeros)
    relus_ones = torch.nn.functional.relu(ones)
    return relus_zeros.mean() + relus_ones.mean()


def loss_spatial_gradient(classif, inputs_unorm, inputs, dmax_grad, safe_ball_size, depth_max):
    """ReLU loss for penalizing large spatial gradients of the network."""
    grad = torch.autograd.grad(classif, inputs, grad_outputs=torch.ones_like(classif), retain_graph=True, create_graph=True)[0]
    grad.requires_grad_(True)
    max_grad = torch.where(torch.linalg.vector_norm(inputs_unorm, dim=1) > 0.5, depth_max/(2*dmax_grad), 50)  # allow steeper close to camera
    norm_grad = torch.linalg.vector_norm(grad, dim=1)
    gradient_loss = torch.nn.functional.relu(norm_grad-max_grad).mean()
    return gradient_loss
