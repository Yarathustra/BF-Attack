import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.bfmifgsm import BF_MIFGSM

import scipy.stats as st
import numpy as np

class BF_TIM(BF_MIFGSM):
    """
    BF-TIM Attack
    Boundary-based Translation-Invariant Momentum Iterative FGSM with multiple boundary point sampling
    Combines momentum from MI-FGSM, boundary sampling from BF-FGSM, and translation-invariance from TIM
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, 
                 kernel_type='gaussian', kernel_size=15, targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', num_boundary_points=20, max_num_move=5, 
                 shrinkage_factor=0.6, initial_direction_std=20, device=None, **kwargs):
        """
        Arguments:
            kernel_type (str): the type of kernel (gaussian/uniform/linear).
            kernel_size (int): the size of kernel.
            num_boundary_points (int): number of boundary points to sample
            max_num_move (int): maximum number of moves to find boundary
            shrinkage_factor (float): shrinkage factor for boundary search
            initial_direction_std (float): standard deviation for initial direction sampling
            decay (float): the decay factor for momentum calculation
        """
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, 
                         num_boundary_points, max_num_move, shrinkage_factor, initial_direction_std, device)
        self.kernel = self.generate_kernel(kernel_type, kernel_size)

    def generate_kernel(self, kernel_type, kernel_size, nsig=3):
        """
        Generate the gaussian/uniform/linear kernel

        Arguments:
            kernel_type (str): the method for initializing the kernel
            kernel_size (int): the size of kernel
        """
        if kernel_type.lower() == 'gaussian':
            x = np.linspace(-nsig, nsig, kernel_size)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        elif kernel_type.lower() == 'uniform':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif kernel_type.lower() == 'linear':
            kern1d = 1 - np.abs(np.linspace((-kernel_size+1)//2, (kernel_size-1)//2, kernel_size)/(kernel_size**2))
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        else:
            raise Exception("Unsupported kernel type {}".format(kernel_type))
        
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for BF-TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad
    
    def forward(self, data, label, **kwargs):
        """
        The BF-TIM attack procedure
        
        Arguments:
            data (N, C, H, W): tensor for input images
            label (N,): tensor for ground-truth labels if untargeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        
        # Initialize momentum
        momentum_grad = torch.zeros_like(data)
        
        for t in range(self.epoch):
            x = data + delta
            grad_boundary = []
            
            # Sample multiple boundary points
            for i in range(self.num_boundary_points):
                # Sample random direction di ~ N(0, σ²I)
                di = torch.randn_like(x) * (self.initial_direction_std / 255.0)
                
                # Get original prediction
                with torch.no_grad():
                    original_output = self.get_logits(x)
                    original_pred = torch.argmax(original_output, dim=1)
                
                # Find boundary point by moving along di
                boundary_x = self.find_boundary_point(x, di, original_pred)
                
                # Calculate gradient at boundary point
                boundary_x.requires_grad_(True)
                boundary_output = self.get_logits(boundary_x)
                boundary_loss = self.get_loss(boundary_output, label)
                
                grad = self.get_grad(boundary_loss, boundary_x)
                grad_boundary.append(grad.detach())
            
            # Average all boundary gradients
            avg_boundary_grad = torch.stack(grad_boundary).mean(dim=0)
            
            # Update momentum
            momentum_grad = self.decay * momentum_grad + avg_boundary_grad / torch.norm(avg_boundary_grad, p=1)
            
            # Update delta using sign of momentum gradient
            delta = self.update_delta(delta, data, momentum_grad, self.alpha)
        
        return delta.detach()