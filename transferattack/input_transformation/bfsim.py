import torch

from ..utils import *
from ..gradient.bfmifgsm import BF_MIFGSM

class BF_SIM(BF_MIFGSM):
    """
    BF-SIM Attack
    Boundary-based Scale Invariance Momentum Iterative FGSM with multiple boundary point sampling
    Combines momentum from MI-FGSM, boundary sampling from BF-FGSM, and scale invariance from SIM
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, 
                 num_scale=5, targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 num_boundary_points=20, max_num_move=5, shrinkage_factor=0.6, 
                 initial_direction_std=20, device=None, **kwargs):
        """
        Arguments:
            num_scale (int): the number of scaled copies in each iteration
            num_boundary_points (int): number of boundary points to sample
            max_num_move (int): maximum number of moves to find boundary
            shrinkage_factor (float): shrinkage factor for boundary search
            initial_direction_std (float): standard deviation for initial direction sampling
            decay (float): the decay factor for momentum calculation
        """
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, 
                         num_boundary_points, max_num_move, shrinkage_factor, initial_direction_std, device)
        self.num_scale = num_scale
    
    def forward(self, data, label, **kwargs):
        """
        The BF-SIM attack procedure
        
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
            
            # For each scale
            for s in range(self.num_scale):
                x_scaled = x / (2 ** s)
                
                # Sample multiple boundary points for this scale
                for i in range(self.num_boundary_points):
                    # Sample random direction di ~ N(0, σ²I)
                    di = torch.randn_like(x_scaled) * (self.initial_direction_std / 255.0)
                    
                    # Get original prediction on scaled input
                    with torch.no_grad():
                        original_output = self.get_logits(x_scaled)
                        original_pred = torch.argmax(original_output, dim=1)
                    
                    # Find boundary point by moving along di
                    boundary_x = self.find_boundary_point(x_scaled, di, original_pred)
                    
                    # Calculate gradient at boundary point
                    boundary_x.requires_grad_(True)
                    boundary_output = self.get_logits(boundary_x)
                    boundary_loss = self.get_loss(boundary_output, label)
                    
                    grad = torch.autograd.grad(boundary_loss, boundary_x, 
                                              retain_graph=False, create_graph=False)[0]
                    grad_boundary.append(grad.detach())
            
            # Average all boundary gradients
            avg_boundary_grad = torch.stack(grad_boundary).mean(dim=0)
            
            # Update momentum
            momentum_grad = self.decay * momentum_grad + avg_boundary_grad / torch.norm(avg_boundary_grad, p=1)
            
            # Update delta using sign of momentum gradient
            delta = self.update_delta(delta, data, momentum_grad, self.alpha)
        
        return delta.detach()