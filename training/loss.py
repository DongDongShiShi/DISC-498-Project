# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        noisy_input = y + n
        D_yn = net(noisy_input, sigma, labels, augment_labels=augment_labels)

        if torch.distributed.get_rank() == 0:
            self.plot_counter += 1   # <-- update counter
            if self.plot_counter % 100 == 0:   # <-- only every 100 times
                import matplotlib.pyplot as plt
                import os
                pred = D_yn.detach().cpu().numpy()
                gt = y.detach().cpu().numpy()
                noisy = noisy_input.detach().cpu().numpy()

                plt.figure(figsize=(8, 4))

                plt.subplot(2, 3, 1)
                plt.title('Pred ch0')
                plt.imshow(pred[0, 0], cmap='viridis')
                plt.axis('off')

                plt.subplot(2, 3, 2)
                plt.title('Pred ch1')
                plt.imshow(pred[0, 1], cmap='viridis')
                plt.axis('off')

                plt.subplot(2, 3, 3)
                plt.title('GT ch0')
                plt.imshow(gt[0, 0], cmap='viridis')
                plt.axis('off')

                plt.subplot(2, 3, 4)
                plt.title('GT ch1')
                plt.imshow(gt[0, 1], cmap='viridis')
                plt.axis('off')

                plt.subplot(2, 3, 5)
                plt.title('Noisy ch0')
                plt.imshow(noisy[0, 0], cmap='viridis')
                plt.axis('off')

                plt.subplot(2, 3, 6)
                plt.title('Noisy ch1')
                plt.imshow(noisy[0, 1], cmap='viridis')
                plt.axis('off')

                plt.tight_layout()

                # Save with a different name every time
                save_path = os.path.join(os.getcwd(), f'training_debug_step_{self.plot_counter:06d}.png')
                plt.savefig(save_path)
                plt.close()
                
        loss = weight * ((D_yn - y) ** 2)
        print(loss.mean().item())
        return loss

#----------------------------------------------------------------------------
