import torch

from ..utils import *
from ..gradient.bfmifgsm import BF_MIFGSM

class BF_Admix(BF_MIFGSM):
    """
    BF-Admix Attack
    Boundary-based Admix Attack combining boundary sampling from BF-FGSM with Admix transformations.
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scaled copies in each iteration.
        num_admix (int): the number of admixed images in each iteration.
        admix_strength (float): the strength of admixed images.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        num_boundary_points (int): number of boundary points to sample.
        max_num_move (int): maximum number of moves to find boundary.
        shrinkage_factor (float): shrinkage factor for boundary search.
        initial_direction_std (float): standard deviation for initial direction sampling.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=5, num_admix=3, admix_strength=0.2,
        num_boundary_points=20, max_num_move=5, shrinkage_factor=0.6, initial_direction_std=20

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfadmix/resnet50 --attack bfadmix --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/bfadmix/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, 
                 num_scale=5, num_admix=3, admix_strength=0.2, targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', num_boundary_points=20, max_num_move=5, 
                 shrinkage_factor=0.6, initial_direction_std=20, device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, 
                         num_boundary_points, max_num_move, shrinkage_factor, initial_direction_std, device)
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.admix_strength = admix_strength
    
    def forward(self, data, label, **kwargs):
        """
        The BF-Admix attack procedure
        
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
        
        N = data.size(0)
        
        for t in range(self.epoch):
            x = data + delta
            grad_boundary = []
            
            # For each scale
            for s in range(self.num_scale):
                # Create admixed images
                admix_images = torch.concat([(x + self.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_admix)], dim=0)
                x_scaled = admix_images / (2 ** s)
                
                # Get original prediction on scaled admixed input
                with torch.no_grad():
                    original_output = self.get_logits(x_scaled)
                    original_pred = torch.argmax(original_output, dim=1)
                
                # Sample multiple boundary points for this scaled admixed input
                for i in range(self.num_boundary_points):
                    # Sample random direction di ~ N(0, σ²I)
                    di = torch.randn_like(x_scaled) * (self.initial_direction_std / 255.0)
                    
                    # Find boundary point
                    boundary_x = self.find_boundary_point(x_scaled, di, original_pred)
                    
                    # Calculate gradient at boundary point
                    boundary_x.requires_grad_(True)
                    boundary_output = self.get_logits(boundary_x)
                    boundary_loss = self.get_loss(boundary_output, label.repeat(self.num_admix))
                    
                    grad = torch.autograd.grad(boundary_loss, boundary_x, 
                                              retain_graph=False, create_graph=False)[0]
                    grad_boundary.append(grad.detach())
            
            # Average all boundary gradients
            avg_boundary_grad = torch.stack(grad_boundary).mean(dim=0)  # (num_admix * N, ...)
            
            # Reshape and average over admix dimension to get (N, ...)
            avg_boundary_grad = avg_boundary_grad.view(self.num_admix, N, *avg_boundary_grad.shape[1:]).mean(dim=0)
            
            # Update momentum
            momentum_grad = self.decay * momentum_grad + avg_boundary_grad / torch.norm(avg_boundary_grad, p=1)
            
            # Update delta using sign of momentum gradient
            delta = self.update_delta(delta, data, momentum_grad, self.alpha)
        
        return delta.detach()