import torch

from ..utils import *
from ..attack import Attack

class BF_MIFGSM(Attack):
    """
    BF-MI-FGSM Attack
    Boundary-based Momentum Iterative FGSM with multiple boundary point sampling
    Combines momentum from MI-FGSM with boundary sampling from BF-FGSM
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, 
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 num_boundary_points=10, max_num_move=20, shrinkage_factor=0.8, 
                 initial_direction_std=1.0, device=None, **kwargs):
        """
        Arguments:
            num_boundary_points (int): number of boundary points to sample
            max_num_move (int): maximum number of moves to find boundary
            shrinkage_factor (float): shrinkage factor for boundary search
            initial_direction_std (float): standard deviation for initial direction sampling
            decay (float): the decay factor for momentum calculation
        """
        super().__init__('BF-MI-FGSM', model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_boundary_points = num_boundary_points
        self.max_num_move = max_num_move
        self.shrinkage_factor = shrinkage_factor
        self.initial_direction_std = initial_direction_std

    def forward(self, data, label, **kwargs):
        """
        The BF-MI-FGSM attack procedure
        
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
    
    def find_boundary_point(self, x, direction, original_pred):
        """
        Find boundary point by binary search along direction
        
        Arguments:
            x: current adversarial example
            direction: search direction
            original_pred: original prediction class
        
        Returns:
            boundary_x: point on decision boundary
        """
        # Initialize rho to move out of decision boundary
        rho = 1.0
        moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
        
        # Move out of original classification region
        with torch.no_grad():
            moved_output = self.get_logits(moved_x)
            moved_pred = torch.argmax(moved_output, dim=1)
        
        num_moves = 0
        while torch.equal(moved_pred, original_pred) and num_moves < self.max_num_move:
            rho *= 2.0
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
            num_moves += 1
        
        # Binary search to find boundary point
        while not torch.equal(moved_pred, original_pred) and rho > 1e-6:
            rho *= self.shrinkage_factor
            moved_x = torch.clamp(x + rho * direction, 0.0, 1.0)
            
            with torch.no_grad():
                moved_output = self.get_logits(moved_x)
                moved_pred = torch.argmax(moved_output, dim=1)
        
        return moved_x.clone().detach()