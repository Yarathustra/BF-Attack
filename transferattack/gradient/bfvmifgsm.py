import torch

from ..utils import *
from ..attack import Attack

class BF_VMIFGSM(Attack):
    """
    BF-VMI-FGSM Attack
    Combining VMI-FGSM with Boundary Fitting by using boundary sampling in variance calculation
    Based on VMI-FGSM and BF-MI-FGSM
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='BF-VMI-FGSM', 
                max_num_move=20, shrinkage_factor=0.8, initial_direction_std=1.0, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon  # Kept for compatibility, but not used in boundary sampling
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor  # Used as number of boundary points
        self.max_num_move = max_num_move
        self.shrinkage_factor = shrinkage_factor
        self.initial_direction_std = initial_direction_std

    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        """
        Calculate the gradient variance using boundary sampling instead of neighborhood sampling
        """
        grad = 0
        x = data + delta
        
        # Get original prediction (computed once per call for efficiency)
        with torch.no_grad():
            original_output = self.get_logits(x)
            original_pred = torch.argmax(original_output, dim=1)
        
        for _ in range(self.num_neighbor):
            # Sample random direction di ~ N(0, σ²I)
            di = torch.randn_like(x) * (self.initial_direction_std / 255.0)
            
            # Find boundary point
            boundary_x = self.find_boundary_point(x, di, original_pred)
            
            # Calculate gradient at boundary point
            boundary_x.requires_grad_(True)
            boundary_output = self.get_logits(boundary_x)
            boundary_loss = self.get_loss(boundary_output, label)
            
            boundary_grad = torch.autograd.grad(boundary_loss, boundary_x, 
                                              retain_graph=False, create_graph=False)[0]
            grad += boundary_grad.detach()
        
        return grad / self.num_neighbor - cur_grad

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

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for BF-VMI-FGSM

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargeted, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum, variance = 0, 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad+variance, momentum)

            # Calculate the variance using boundary sampling
            variance = self.get_variance(data, delta, label, grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()