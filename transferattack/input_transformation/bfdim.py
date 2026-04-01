import torch
import torch.nn.functional as F

from ..utils import *
from ..gradient.bfmifgsm import BF_MIFGSM

class BF_DIM(BF_MIFGSM):
    """
    BF-DIM Attack
    Boundary-based Diverse Input Momentum Iterative FGSM with multiple boundary point sampling
    Combines momentum from MI-FGSM, boundary sampling from BF-FGSM, and input diversity from DIM
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.0, 
                 resize_rate=1.1, diversity_prob=0.5, targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', num_boundary_points=20, max_num_move=5, 
                 shrinkage_factor=0.6, initial_direction_std=20, device=None, **kwargs):
        """
        Arguments:
            resize_rate (float): the relative size of the resized image
            diversity_prob (float): the probability for transforming the input image
            num_boundary_points (int): number of boundary points to sample
            max_num_move (int): maximum number of moves to find boundary
            shrinkage_factor (float): shrinkage factor for boundary search
            initial_direction_std (float): standard deviation for initial direction sampling
            decay (float): the decay factor for momentum calculation
        """
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, 
                         num_boundary_points, max_num_move, shrinkage_factor, initial_direction_std, device)
        if resize_rate < 1:
            raise Exception("Error! The resize rate should be larger than 1.")
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
    
    def transform(self, x, **kwargs):
        """
        Random transform the input images
        """
        # do not transform the input image
        if torch.rand(1) > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    
    def forward(self, data, label, **kwargs):
        """
        The BF-DIM attack procedure
        
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
            # Apply input diversity transformation
            x_transformed = self.transform(x)
            grad_boundary = []
            
            # Sample multiple boundary points
            for i in range(self.num_boundary_points):
                # Sample random direction di ~ N(0, σ²I)
                di = torch.randn_like(x_transformed) * (self.initial_direction_std / 255.0)
                
                # Get original prediction on transformed input
                with torch.no_grad():
                    original_output = self.get_logits(x_transformed)
                    original_pred = torch.argmax(original_output, dim=1)
                
                # Find boundary point by moving along di
                boundary_x = self.find_boundary_point(x_transformed, di, original_pred)
                
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