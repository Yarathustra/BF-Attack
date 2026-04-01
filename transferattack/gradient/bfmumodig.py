import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import scipy.stats as st
from .mumodig import MUMODIG
class BF_MUMODIG(MUMODIG):
    """
    BF-MUMODIG Attack
    Boundary Fitting MUMODIG Attack, combining MUMODIG with boundary sampling from BF-MIFGSM

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        N_trans (int): the number of total auxiliary inputs
        N_base (int): baseline number
        N_intepolate (int): interpolation point number
        region_num (int): the region number
        lamb (float): the position factor
        num_boundary_points (int): number of boundary points to sample
        max_num_move (int): maximum number of moves to find boundary
        shrinkage_factor (float): shrinkage factor for boundary search
        initial_direction_std (float): standard deviation for initial direction sampling
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, N_trans=6, N_base=1, N_intepolate=1, region_num=2, lamb=0.65
        num_boundary_points=10, max_num_move=20, shrinkage_factor=0.8, initial_direction_std=1.0

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data5/bfmumodig/resnet50 --attack bfmumodig --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data5/bfmumodig/resnet50 --eval    
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., 
                 N_trans=6, N_base=1, N_intepolate=1, region_num=2, lamb=0.65,
                 num_boundary_points=10, max_num_move=20, shrinkage_factor=0.8, initial_direction_std=1.0,
                 targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None, attack='BF-MUMODIG', **kwargs): 

        super().__init__(model_name, epsilon, alpha, epoch, decay, N_trans, N_base, N_intepolate, region_num, lamb,
                         targeted, random_start, norm, loss, device, attack)
        self.num_boundary_points = num_boundary_points
        self.max_num_move = max_num_move
        self.shrinkage_factor = shrinkage_factor
        self.initial_direction_std = initial_direction_std

    def ig(self, data, delta, label, **kwargs): 
        
        ig = 0

        for i_base in range(self.N_base):
            baseline = self.quant(data+delta).clone().detach().to(self.device) 
            path = data+delta - baseline
            acc_grad = 0   
            for i_inter in range(self.N_intepolate):

                x_interplotate = baseline + (i_inter + self.lamb) * path / self.N_intepolate 
                
                # Boundary fitting: sample boundary points and average gradients
                grad_boundary = []
                for i in range(self.num_boundary_points):
                    di = torch.randn_like(x_interplotate) * (self.initial_direction_std / 255.0)
                    boundary_x = self.find_boundary_point(x_interplotate, di, label)
                    boundary_x.requires_grad_(True)
                    logits = self.get_logits(boundary_x)
                    loss = self.get_loss(logits, label)
                    grad = torch.autograd.grad(loss, boundary_x, retain_graph=False, create_graph=False)[0]
                    grad_boundary.append(grad.detach())
                each_ig_grad = torch.stack(grad_boundary).mean(dim=0)
            
                # accumulate grads
                acc_grad += each_ig_grad 
            ig += acc_grad * path

        return ig

    def exp_ig(self, data, delta, label, **kwargs):
        
        ig = 0

        for i_trans in range(self.N_trans):

            x_transform = self.select_transform_apply(data+delta)

            for i_base in range(self.N_base):

                baseline = self.quant(x_transform).clone().detach().to(self.device) # quant baseline

                path = x_transform - baseline 
                
                acc_grad = 0            
                for i_inter in range(self.N_intepolate):

                    x_interplotate = baseline + (i_inter + self.lamb) / self.N_intepolate * path  

                    # Boundary fitting: sample boundary points and average gradients
                    grad_boundary = []
                    for i in range(self.num_boundary_points):
                        di = torch.randn_like(x_interplotate) * (self.initial_direction_std / 255.0)
                        boundary_x = self.find_boundary_point(x_interplotate, di, label)
                        boundary_x.requires_grad_(True)
                        logits = self.get_logits(boundary_x)
                        loss = self.get_loss(logits, label)
                        grad = torch.autograd.grad(loss, boundary_x, retain_graph=False, create_graph=False)[0]
                        grad_boundary.append(grad.detach())
                    each_ig_grad = torch.stack(grad_boundary).mean(dim=0)

                    acc_grad += each_ig_grad 

                ig += acc_grad * path  

        return ig

    def find_boundary_point(self, x, direction, label):
        """
        Find boundary point by binary search along direction
        
        Arguments:
            x: current point
            direction: search direction
            label: target label for targeted, or original label for untargeted
        
        Returns:
            boundary_x: point on decision boundary
        """
        # Get original prediction
        with torch.no_grad():
            original_output = self.get_logits(x)
            original_pred = torch.argmax(original_output, dim=1)
        
        # Initialize rho to move out of decision boundary
        rho = 1.0
        moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
        
        # Move out of original classification region
        with torch.no_grad():
            moved_output = self.get_logits(moved_x)
            moved_pred = torch.argmax(moved_output, dim=1)
        
        if self.targeted:
            condition = torch.equal(moved_pred, label)
        else:
            condition = torch.equal(moved_pred, original_pred)
        
        num_moves = 0
        while condition and num_moves < self.max_num_move:
            rho *= 2.0
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
            if self.targeted:
                condition = torch.equal(moved_pred, label)
            else:
                condition = torch.equal(moved_pred, original_pred)
            num_moves += 1
        
        # Binary search to find boundary point
        while (not condition) and rho > 1e-6:
            rho *= self.shrinkage_factor
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
            if self.targeted:
                condition = torch.equal(moved_pred, label)
            else:
                condition = torch.equal(moved_pred, original_pred)
        
        return moved_x.clone().detach()